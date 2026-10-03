""" RBF Your SDF: reconstruct a surface from signed distance samples.

A thin wrapper around the C++ module ``rbfyoursdf_cpp``. ``main_algorithm`` takes
keyword arguments and builds the C++ ``Options`` from them; the bindings are
re-exported for direct use.
"""

import warnings

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


def main_algorithm(sdf_points, sdf_values, *,
                   max_iters=10, clamp=True, turn_off_short_arcs=False, reg=0,
                   interpolator_type='PU', interp_partition='sphere', interp_overlap=0.2,
                   pair_local=False, iter_gradient_finding='optimize', grad_optimizer='bfgs',
                   lr=0.2, optim_steps=5, degen_tol=1e-5, gt_gradients=None, verbose=True,
                   name='default', grid_len=20, export_projections=False, export_short_arcs=False,
                   return_options=False):
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

    Returns
    -------
    MainResult with projections (N, 3), visibility_mask (N,) and the fitted
    interpolator (with predict() and extract_surface()); or (MainResult, Options)
    if return_options.
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
    result = rbfyoursdf_cpp.main_algorithm(points, values, opts)
    return (result, opts) if return_options else result
