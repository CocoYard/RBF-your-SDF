""" RBF Your SDF: reconstruct a surface from signed distance samples.

A thin wrapper around the C++ module ``rbfyoursdf_cpp``. ``main_algorithm`` takes
keyword arguments and builds the C++ ``Options`` from them; by default it also
normalizes the input to a centered unit box, since the C++ tolerances are absolute
lengths tuned at that scale. The bindings are re-exported for direct use.
"""

import warnings
from typing import Literal, overload

import numpy as np

from . import rbfyoursdf_cpp
from .rbfyoursdf_cpp import (Options, Tolerance, MainResult, Interpolator,
                             DuchonInterpolator, PUInterpolator, are_points_visible,
                             has_openmp)

__all__ = ['main_algorithm', 'rbfyoursdf_cpp', 'Options', 'Tolerance', 'MainResult',
           'Interpolator', 'DuchonInterpolator', 'PUInterpolator', 'are_points_visible',
           'has_openmp']

if not has_openmp():
    warnings.warn(
        "rbfyoursdf was built without OpenMP and runs single-threaded (much slower). "
        "For multithreading, install OpenMP (macOS: `brew install libomp`) and reinstall "
        "with `pip install --force-reinstall --no-cache-dir`.", stacklevel=2)


class ScaledInterpolator:
    """ An interpolator fitted in normalized coordinates, queried in input coordinates.

    The C++ interpolator was fitted to x' = (x - center) / scale with values
    d' = d / scale, so f(x) = scale * f'(x') and grad f(x) = grad f'(x').
    """

    def __init__(self, interpolator, center, scale):
        self.interpolator = interpolator  # the C++ interpolator, in normalized coordinates
        self.center = center
        self.scale = scale

    def _to_unit(self, x):
        return (np.asarray(x, dtype=np.float64) - self.center) / self.scale

    @property
    def verbose(self):
        return self.interpolator.verbose

    @verbose.setter
    def verbose(self, value):
        self.interpolator.verbose = value

    def predict(self, x_new, chunk_size=None):
        """ Signed distance at x_new (M, 3), shape (M, 1). """
        args = () if chunk_size is None else (chunk_size,)
        return self.scale * self.interpolator.predict(self._to_unit(x_new), *args)

    def predict_gradients(self, x_new, chunk_size=None):
        """ Gradient of the signed distance at x_new (M, 3), shape (M, 3). """
        args = () if chunk_size is None else (chunk_size,)
        return self.interpolator.predict_gradients(self._to_unit(x_new), *args)

    def extract_surface(self, bbox_min, bbox_max, nx, ny, nz, iso=0.0, chunk_size=5000,
                        lipschitz_postfix=True, use_dual_contouring=False):
        """ Extract the iso level set inside [bbox_min, bbox_max] on an nx*ny*nz grid.
            Returns (V, F) with V in input coordinates. """
        V, F = self.interpolator.extract_surface(
            self._to_unit(bbox_min), self._to_unit(bbox_max), nx, ny, nz,
            iso=iso / self.scale, chunk_size=chunk_size,
            lipschitz_postfix=lipschitz_postfix, use_dual_contouring=use_dual_contouring)
        return np.asarray(V) * self.scale + self.center, F


class Result:
    """ main_algorithm's result in input coordinates.

    projections: (N, 3) tangent point of every sample.
    visibility_mask: (N,) 1 = visible, 0 = occluded.
    interpolator: ScaledInterpolator with predict(), predict_gradients() and extract_surface().
    center, scale: the normalization, x' = (x - center) / scale.
    """

    def __init__(self, result, center, scale):
        self.projections = np.asarray(result.projections) * scale + center
        self.visibility_mask = np.asarray(result.visibility_mask)
        self.interpolator = ScaledInterpolator(result.interpolator, center, scale)
        self.center = center
        self.scale = scale


@overload
def main_algorithm(sdf_points, sdf_values, *, return_options: Literal[False] = False,
                   **kwargs) -> Result: ...
@overload
def main_algorithm(sdf_points, sdf_values, *, return_options: Literal[True],
                   **kwargs) -> tuple[Result, Options]: ...
def main_algorithm(sdf_points, sdf_values, *,
                   max_iters=10, clamp=True, turn_off_short_arcs=False, reg=0,
                   interpolator_type='PU', interp_partition='sphere', interp_overlap=0.2,
                   pair_local=False, iter_gradient_finding='optimize', grad_optimizer='bfgs',
                   lr=0.2, optim_steps=5, degen_tol=1e-5, gt_gradients=None, verbose=True,
                   name='default', grid_len=20, export_projections=False, export_short_arcs=False,
                   return_options=False, normalize=True):
    """ Estimate the tangent point of every SDF sample and fit an interpolator to them.

    Parameters
    ----------
    sdf_points: (N, 3) sample positions.
    sdf_values: (N,) signed distances at sdf_points (negative inside).
    max_iters: outer iterations of gradient finding and refitting.
    clamp: clamp projections to the nearest visible arc point.
    turn_off_short_arcs: do not use short-arc midpoints as surface candidates.
    reg: RBF regularization.
    interpolator_type: 'PU' (partition of unity) or 'Duchon'.
    interp_partition: 'box' or 'sphere', the shape of a PU patch.
    interp_overlap: PU patch overlap.
    pair_local: PU: pair each local RBF solve with missing input/projection partners.
    iter_gradient_finding: 'optimize' or 'sample'.
    grad_optimizer: solver behind 'optimize': 'bfgs' (batched BFGS), 'ascent'
        (fixed-step projected gradient ascent) or 'lbfgspp' (one LBFGS++ solve per point).
    lr: step size for 'ascent'.
    optim_steps: optimizer steps per point in each outer iteration.
    degen_tol: total exposed-arc length below which a sphere's exposed region
        collapses to a tangent point.
    gt_gradients: optional (N, 3) gradients; if given, skip the iteration and fit
        once with them.
    verbose: print progress.
    name, grid_len: only used in log output and export filenames.
    export_projections: write PLY files of the projections to out/<name>/.
    export_short_arcs: write a PLY file of the short-arc points to out/<name>/.
    return_options: also return the C++ Options, whose degenerate_points and
        short_arc_candidates are filled in by the run.
    normalize: run on the input translated and scaled so that its bounding box is
        centered at the origin with longest side spanning [-0.5, 0.5]. The
        Options returned by return_options and the PLY exports stay in it too.

    Returns
    -------
    Result with projections (N, 3), visibility_mask (N,) and the fitted
    interpolator (with predict() and extract_surface()), all in input
    coordinates; or (Result, Options) if return_options.
    """
    opts = rbfyoursdf_cpp.Options()
    opts.max_iters = max_iters
    opts.clamp = clamp
    opts.turn_off_short_arcs = turn_off_short_arcs
    opts.reg = reg
    opts.interpolator_type = interpolator_type
    opts.interp_partition = interp_partition
    opts.interp_overlap = interp_overlap
    opts.pair_local = pair_local
    opts.iter_gradient_finding = iter_gradient_finding
    opts.grad_optimizer = grad_optimizer
    opts.lr = lr
    opts.optim_steps = optim_steps
    opts.degen_tol = degen_tol
    if gt_gradients is not None:
        opts.gt_gradients = np.asarray(gt_gradients, dtype=np.float64)
    opts.verbose = verbose
    opts.name = name
    opts.grid_len = grid_len
    opts.export_projections = export_projections
    opts.export_short_arcs = export_short_arcs

    points = np.asarray(sdf_points, dtype=np.float64)
    values = np.asarray(sdf_values, dtype=np.float64).ravel()
    center, scale = np.zeros(3), 1.0
    if normalize:
        lo, hi = points.min(axis=0), points.max(axis=0)
        center = (lo + hi) / 2
        scale = float(np.max(hi - lo)) or 1.0  # a single point has zero extent
        points = (points - center) / scale
        values = values / scale

    result = Result(rbfyoursdf_cpp.main_algorithm(points, values, opts), center, scale)
    return (result, opts) if return_options else result
