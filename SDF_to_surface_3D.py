import os
import trimesh
try:
    import gpytoolbox as gpy
except ModuleNotFoundError:  # no cp314 wheel; reach_for_the_arcs/PSR paths unavailable
    gpy = None
import igl
import numpy as np
import time
from enum import Enum
from util import mesh_distances

# Seed for all randomness in generate_test_mesh_data (scatter sampling, noise).
# __main__ overrides this; importers can set it via `sdf3d.seed = ...` like data_dir.
# None = nondeterministic.
seed = None

class Options:
    def __init__(self, grid_len=20, clamp=True, max_iters=10, name='horse', lr=0.2, optim_steps=5,
                 turn_off_short_arcs=False, export_short_arcs=False, export_projections=False, reg=0,
                 use_gt_gradients=False, interpolator_type='PU', interp_partition='sphere', overlap=0.2, cpp_dc=True,
                 post_processing=False, iter_gradient_finding='optimize', verbose=True,
                pair_local=False, noise=0, bound=1, scatter=False,
                grad_optimizer='bfgs', extract_padding=0.0, degen_tol=1e-5):
        self.grid_len = grid_len
        self.max_iters = max_iters
        self.clamp = clamp
        self.turn_off_short_arcs = turn_off_short_arcs
        self.name = name
        self.path_to_obj = f'{data_dir}/{name}.obj'
        self.export_short_arcs = export_short_arcs  # write a .ply of the short-arc (degenerate) points to out/<name>/
        self.export_projections = export_projections  # write .ply files of the visible / invisible projections to out/<name>/
        self.use_gt_gradients = use_gt_gradients  # skip gradient estimation: fit once with the ground-truth gradients
        self.interpolator_type = interpolator_type  # 'Duchon' or 'PU'
        self.interp_partition = interp_partition  # 'box' or 'sphere', only for PU interpolator. Meaning the shape of the patch.
        self.interp_overlap = overlap
        self.pair_local = pair_local  # PU: pair each local RBF solve with missing input/projection partners without change to the patch size.
        self.post_processing = post_processing  # Lipschitz post-fix during surface extraction
        self.reg = reg  # RBF regularization
        self.iter_gradient_finding = iter_gradient_finding  # 'optimize' or 'sample'
        self.cpp_dc = cpp_dc  # extract with dual contouring (else marching cubes)
        self.verbose = verbose
        self.lr = lr
        self.optim_steps = optim_steps  # optimizer steps per point in each outer iteration
        # solver behind iter_gradient_finding='optimize':
        # 'ascent' = fixed-step projected gradient ascent (lr is its step size),
        # 'lbfgspp' = one LBFGS++ solve per point
        # 'bfgs' = batched BFGS
        self.grad_optimizer = grad_optimizer
        self.extract_padding = extract_padding  # the padding added to the bounding box when extracting the zero level set, to avoid cutting off the surface
        # Total exposed-arc length (a length, not an angle) below which a sphere's
        # exposed region collapses to a tangent point, making the short-arc
        # midpoint a surface candidate. Swept by additional_experiments/degen_tol.py.
        self.degen_tol = degen_tol

        # Filled from generate_test_mesh_data by test_our_method / get_tangent_points;
        # only used when use_gt_gradients.
        self.gt_gradients = None

        self.noise = noise
        self.bound = bound
        self.scatter = scatter

    def print(self):
        # Every setting, so a new field cannot be left out. Runtime state filled in
        # by the runners (arrays, the C++ Options) is not a setting and is skipped.
        runtime = ('gt_gradients', 'cpp_options')
        print("Options: " + ", ".join(f"{k}={v}" for k, v in vars(self).items()
                                      if k not in runtime))

def generate_test_mesh_data( path_to_mesh, outbase, grid_len=10, save=False, noise=0.0, bound=1.0, scatter=False ):
    '''
    Normalize the mesh at path_to_mesh to the unit cube and sample its exact SDF on a
    grid_len^3 grid over the padded bounding box (or at uniform random positions if
    scatter), optionally with Gaussian noise on the distances and truncated to |d| <= bound.
    Returns:
    mesh:      the normalized trimesh (ground truth for error evaluation)
    points:    (N, 3) sample positions
    distances: (N,)   signed distances (negative inside)
    gradients: (N, 3) unit ground-truth gradients
    '''
    # All random draws below (scatter points, noise) come from this generator,
    # seeded by the module-level `seed` (set in __main__, like data_dir).
    # seed=None keeps the old nondeterministic behavior for importers.
    rng = np.random.default_rng(seed)

    # Load the mesh
    mesh = trimesh.load(path_to_mesh)
    # Normalize the mesh to fit within a unit cube
    min = np.min( mesh.vertices, axis=0 )
    max = np.max( mesh.vertices, axis=0 )
    mesh.vertices -= (min + max) / 2
    mesh.vertices /= np.max( max - min )

    # Generate equally spaced points around the mesh bounding box
    bbox_min = np.min(mesh.vertices, axis=0) - 0.1
    bbox_max = np.max(mesh.vertices, axis=0) + 0.1
    x = np.linspace(bbox_min[0], bbox_max[0], grid_len)
    y = np.linspace(bbox_min[1], bbox_max[1], grid_len)
    z = np.linspace(bbox_min[2], bbox_max[2], grid_len)
    print("bbox_min:", bbox_min, "bbox_max:", bbox_max)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    if scatter:
        # totally random points in the bounding box
        points = rng.uniform(bbox_min, bbox_max, (grid_len**3, 3))
    # Find the closest points on the mesh surface
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    sq_dists, _, closest = igl.point_mesh_squared_distance(points, V, F)
    distances = np.sqrt(sq_dists)

    gradients = points - closest
    # Normalize gradients
    norm_temp = np.linalg.norm(gradients, axis=1, keepdims=True)
    # Avoid division by zero
    norm_temp[np.abs(norm_temp) <= 1e-8] = 1.0
    gradients /= norm_temp

    # Add noise to the distances
    if noise > 0:
        distances += rng.normal(0, noise, distances.shape)

    # Drop samples lying on the surface (|d| <= 1e-8): their gradient is undefined
    mask = np.abs(distances) > 1e-8
    points = points[mask]
    distances = distances[mask]
    gradients = gradients[mask]

    # Use winding number to determine inside/outside
    W = igl.winding_number(V, F, points)
    mask = W > 0.5  # Points with winding number > 0.5 are inside
    distances[mask] *= -1.0  # Invert distances for points inside the mesh
    gradients[mask] *= -1.0  # Invert gradients for points inside the mesh

    if bound < 1.0:
        # Filter points based on SDF values to keep only those within the specified bound
        # (marching cubes is the exception: it cannot read a point set with
        # holes, so _common._mc_samples clamps to +/-bound for MC alone.)
        mask = np.abs(distances) <= bound
        points = points[mask]
        distances = distances[mask]
        gradients = gradients[mask]

    # save to file for reuse
    if save:
        import os
        os.makedirs('normalized_examples', exist_ok=True)
        mesh.export(f'normalized_examples/{outbase}.obj')
        print(f"Saved normalized mesh to normalized_examples/{outbase}.obj")

    return mesh, points, distances, gradients

def test_rfta(options, save_gtmesh=False, screening_weight=10, parallel=True, force_cpu=False, sdf=None):
    """ Reach for the Arcs baseline: reconstruct from the SDF samples and export to out/<name>/. """
    grid_len, path_to_obj = options.grid_len, options.path_to_obj
    if sdf is not None:
        points, distances = sdf
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        _, points, distances, _ = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh, noise=options.noise, bound=options.bound)
    # Export meshes to out/
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    timer = time.perf_counter()
    Vr, Fr = gpy.reach_for_the_arcs(points, distances, screening_weight=screening_weight, parallel=parallel, force_cpu=force_cpu)
    print(f"  ⏱  {'RFTA reconstruction':<30} {time.perf_counter() - timer:>7.2f} s")
    rfta = trimesh.Trimesh(vertices=Vr, faces=Fr)
    # Keep only components whose mean coordinates are fully inside the input bbox and 
    # percent of coordinates inside the input bbox is at least 50%, so PSR "bubble"
    # artifacts that wrap outside the sample region get dropped.
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    components = rfta.split(only_watertight=False)
    kept = [c for c in components
            if np.all(np.mean(c.vertices, axis=0) >= bbox_min)
                and np.all(np.mean(c.vertices, axis=0) <= bbox_max) and np.mean(np.all((c.vertices >= bbox_min) & (c.vertices <= bbox_max), axis=1)) > 0.5]
    if not kept:
        # First retry with a padded bbox — components that just barely poke
        # outside the sample region are usually still legitimate.
        pad = 0.1 * (bbox_max - bbox_min)
        pmin, pmax = bbox_min - pad, bbox_max + pad
        kept = [c for c in components
                if np.all(np.mean(c.vertices, axis=0) >= pmin)
                    and np.all(np.mean(c.vertices, axis=0) <= pmax) and np.mean(np.all((c.vertices >= pmin) & (c.vertices <= pmax), axis=1)) > 0.5]
    if not kept:
        # Last resort: every component crosses even the padded bbox. Keep the
        # largest so we still write a non-empty .obj.
        kept = [max(components, key=lambda m: len(m.faces))]
    filtered = trimesh.util.concatenate(kept)
    if options.noise > 0:
        fname = f'rfta_{grid_len}_noise{options.noise}.obj'
    elif options.bound < 1.0:
        fname = f'rfta_{grid_len}_bound{options.bound}.obj'
    else:
        fname = f'rfta_{grid_len}.obj'
    filtered.export(f'{out_dir}/' + fname)
    print(f"Exported: {out_dir}/" + fname + f"  (kept {len(kept)}/{len(components)} components, {len(filtered.faces)} faces out of {len(Fr)})")

def _mes_dir():
    """ Checkout of maxkohlbrenner/maximal-empty-spheres: $MES_DIR, else third_party/ in this repo. """
    return os.environ.get('MES_DIR') or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'third_party', 'maximal-empty-spheres')

def _import_mes():
    """ Import the optional MES baseline, or raise ImportError saying how to install it.
        MESReconstruction itself only prints an error and returns False when its C++
        executable is missing, so check for the executable here too. """
    import sys
    mes_dir = _mes_dir()
    hint = (f"MES baseline not installed (looked in {mes_dir}). It is optional and only "
            f"needed for the MES comparison; see 'Optional: MES baseline' in "
            f"additional_experiments/README.md, "
            f"or set MES_DIR to an existing checkout.")
    if not os.path.isfile(os.path.join(mes_dir, 'cgal', 'build', 'empty_spheres_reconstruction')):
        raise ImportError(hint)
    if mes_dir not in sys.path:
        sys.path.insert(0, mes_dir)
    try:
        from cgal.EmptySpheresReconstruction import MESReconstruction
    except ImportError as e:
        raise ImportError(hint) from e
    return MESReconstruction

def mes_available():
    """ True if the optional MES baseline is installed, so callers can skip it otherwise. """
    try:
        _import_mes()
        return True
    except ImportError:
        return False

def test_mes(options, save_gtmesh=False, screening_weight=10, sdf=None):
    """ Maximal Empty Spheres baseline: reconstruct from the SDF samples and export to out/<name>/. """
    grid_len, path_to_obj = options.grid_len, options.path_to_obj
    if sdf is not None:
        points, distances = sdf
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        _, points, distances, _ = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh, bound=options.bound)
    MESReconstruction = _import_mes()
    # Export meshes to out/
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    timer = time.perf_counter()
    R_cgal = MESReconstruction(points, distances,screening_weight=screening_weight)
    print(f"  ⏱  {'MES reconstruction':<30} {time.perf_counter() - timer:>7.2f} s")

    fname = f'mes_{grid_len}.obj'
    if options.bound < 1.0:
        fname = f'mes_{grid_len}_bound{options.bound}.obj'
    trimesh.Trimesh(vertices=R_cgal[0], faces=R_cgal[1]).export(f"{out_dir}/" + fname)

    print(f"Exported: {out_dir}/" + fname)

def _import_sdf_cpp():
    """ The compiled C++ module, built into cpp/build (see cpp/CMakeLists.txt). """
    import sys
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpp', 'build')
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import sdf_cpp
    return sdf_cpp

def _build_cpp_options(options : Options):
    """ Translate a Python Options into a C++ sdf_cpp.Options and return it. """
    cpp_opts = _import_sdf_cpp().Options()
    cpp_opts.grid_len = options.grid_len
    cpp_opts.max_iters = options.max_iters
    cpp_opts.clamp = options.clamp
    cpp_opts.reg = options.reg
    cpp_opts.turn_off_short_arcs = options.turn_off_short_arcs
    cpp_opts.interpolator_type = options.interpolator_type
    cpp_opts.interp_partition = options.interp_partition
    cpp_opts.interp_overlap = options.interp_overlap
    cpp_opts.pair_local = options.pair_local
    cpp_opts.name = options.name
    cpp_opts.export_projections = options.export_projections
    cpp_opts.export_short_arcs  = options.export_short_arcs
    cpp_opts.iter_gradient_finding = options.iter_gradient_finding
    cpp_opts.grad_optimizer = options.grad_optimizer
    cpp_opts.lr = options.lr
    cpp_opts.optim_steps = options.optim_steps
    cpp_opts.degen_tol = options.degen_tol
    cpp_opts.verbose = options.verbose
    if options.use_gt_gradients:
        cpp_opts.gt_gradients = options.gt_gradients
    # Keep the C++ Options reachable from the Python one: main_algorithm fills
    # its degenerate_pts in place, so this is how a caller reads back which
    # short-arc candidates survived the filter (degen_tol.py does).
    options.cpp_options = cpp_opts
    return cpp_opts

def _adaptive_resolution(points, grid_len):
    """ Dual-contouring grid resolution: ~target_cells_per_hint cells between adjacent SDF
        samples, clamped to [64, 512]. (The bbox extent cancels out, so this reduces to
        ~4*(grid_len-1).) """
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    hint_spacing = extent / max(grid_len - 1, 1)
    target_cells_per_hint = 4
    return int(np.clip(np.ceil(extent / (hint_spacing / target_cells_per_hint)), 64, 512))

def test_our_method(options : Options, save_gtmesh=False):
    """ Run our pipeline (C++ sdf_cpp) on the SDF samples and export the reconstruction to out/<name>/. """
    grid_len = options.grid_len
    path_to_obj = options.path_to_obj
    iters = options.max_iters
    options.print()
    base_name = path_to_obj.split('/')[-1].split('.')[0]
    mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh, noise=options.noise, bound=options.bound, scatter=options.scatter)
    options.gt_gradients = gt_gradients

    # Create and fit the interpolator
    timer = time.perf_counter()
    result = _import_sdf_cpp().main_algorithm(points, distances, _build_cpp_options(options))
    _vis = np.asarray(result.visibility_mask).ravel()
    if options.verbose:
        print(f"Final visibility: {int((_vis != 0).sum())}/{len(_vis)} ({100.0 * (_vis != 0).mean():.2f}%)")
    print(f"  ⏱  {'Interpolator fitted':<30} {time.perf_counter() - timer:>7.2f} s")

    # ── surface extraction (dual contouring, optional Lipschitz post-fix) ──
    timer = time.perf_counter()
    bbox_min = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 2].min()], dtype=np.float64) - options.extract_padding
    bbox_max = np.array([points[:, 0].max(), points[:, 1].max(), points[:, 2].max()], dtype=np.float64) + options.extract_padding
    resolution = _adaptive_resolution(points, options.grid_len)
    print(f"Grid resolution for surface extraction: {resolution}")

    result.interpolator.verbose = options.verbose
    verts, faces = result.interpolator.extract_surface(
        bbox_min=bbox_min, bbox_max=bbox_max,
        nx=resolution, ny=resolution, nz=resolution, iso=0.0, chunk_size=5000,
        lipschitz_postfix=options.post_processing,
        use_dual_contouring=options.cpp_dc)
    print(f"  ⏱  {'Grid evaluation':<30} {time.perf_counter() - timer:>7.2f} s")
    # Export meshes to out/
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    recon = trimesh.Trimesh(vertices=verts, faces=faces)
    clamp_str = '_clamp' if options.clamp else ''
    post_str = '_post' if options.post_processing else ''
    pair_str = '_pairLocal' if options.pair_local and options.interpolator_type == 'PU' else ''
    if options.cpp_dc:
        post_str = post_str + '_dc'
    else:
        post_str = post_str + '_mc'
    short_arc_str = 'noShortArcs' if options.turn_off_short_arcs else 'shortArcs'
    reg_str = ''
    if options.reg != 0:
        reg_str = f'_reg{options.reg}'
    if options.noise > 0:
        fname = f'ours_{grid_len}_{iters}_{short_arc_str}_{options.interpolator_type}{pair_str}{post_str}_noise{options.noise}'
    elif options.bound < 1.0:
        fname = f'ours_{grid_len}_{iters}_{short_arc_str}_{options.interpolator_type}{pair_str}{post_str}_bound{options.bound}'
    elif options.scatter:
        fname = f'ours_{grid_len}_{iters}_{short_arc_str}_{options.interpolator_type}{pair_str}{post_str}_scatter'
    else:
        fname = f'ours_{grid_len}_{iters}_{short_arc_str}_{options.interpolator_type}{clamp_str}{pair_str}{post_str}_{options.grad_optimizer}'
    fname += reg_str+'.obj'
    recon.export(f'{out_dir}/{fname}')
    print(f"Exported: {out_dir}/{fname}")
    mesh_distances(recon, mesh, verbose=True)
    return points, distances

def test_mc(options : Options, save_gtmesh=False, sdf=None):
    """ Marching cubes directly on the (grid) SDF samples, no interpolation; export to out/<name>/. """
    grid_len = options.grid_len
    path_to_obj = options.path_to_obj
    options.print()
    if sdf is not None:
        points, distances = sdf
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        _, points, distances, _ = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh)
    # Recover the grid structure from the sample coordinates; no interpolation.
    xs = np.unique(np.round(points[:, 0], 8))
    ys = np.unique(np.round(points[:, 1], 8))
    zs = np.unique(np.round(points[:, 2], 8))
    nx, ny, nz = len(xs), len(ys), len(zs)
    ix = np.searchsorted(xs, np.round(points[:, 0], 8))
    iy = np.searchsorted(ys, np.round(points[:, 1], 8))
    iz = np.searchsorted(zs, np.round(points[:, 2], 8))
    grid_values_direct = np.ones((nx, ny, nz))  # missing samples count as outside (+1)
    grid_values_direct[ix, iy, iz] = distances
    sp = ((xs[-1]-xs[0])/(nx-1), (ys[-1]-ys[0])/(ny-1), (zs[-1]-zs[0])/(nz-1))
    from skimage.measure import marching_cubes
    verts2, faces2, _, _ = marching_cubes(grid_values_direct, level=0.0, spacing=sp)
    verts2 += np.array([xs[0], ys[0], zs[0]])
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    trimesh.Trimesh(vertices=verts2, faces=faces2).export(f'{out_dir}/sample_points_{grid_len}.obj')
    print(f"Exported: {out_dir}/sample_points_{grid_len}.obj")

class TangentPoints(Enum):
    GT = 'gt'
    OURS = 'ours'
    RFTA = 'rfta'
    MES = 'mes'

def get_tangent_points(options : Options, method, save_gtmesh=False, screening_weight=10):
    """ Get the tangent points (surface contact points) for the given options and method.

        Returns
        -------
        tangent_pts: (M, 3) array of points lying on the reconstructed surface.
            For OURS the tangent points are the SDF-sample projections and are 1:1 with
            ``points``; for RFTA/MES they are the method's reconstructed point cloud and
            need not be 1:1 with the input samples.
        points:    (N, 3) input SDF sample coordinates.
        distances: (N,)   signed distances at ``points``.
    """
    grid_len = options.grid_len
    path_to_obj = options.path_to_obj
    options.print()
    base_name = path_to_obj.split('/')[-1].split('.')[0]
    mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh)
    options.gt_gradients = gt_gradients
    if method == TangentPoints.OURS:
        # Iterative projection (C++ pipeline, same as test_our_method): tangent points are
        # the SDF-sample projections onto the surface (points - distance * gradient). The
        # optimization's gradients exist only to produce these projections; some come out
        # non-visible / unreliable and the optimization excludes them via a mask when
        # fitting. We mirror that by marking the invalid projections NaN so the
        # reconstruction drops them. The valid tangent points stay index-aligned with
        # points/distances (1:1).
        result = _import_sdf_cpp().main_algorithm(points, distances, _build_cpp_options(options))
        tangent_pts = np.array(result.projections, dtype=np.float64)  # copy: pybind view is read-only
        vis = np.asarray(result.visibility_mask).reshape(-1).astype(bool)
        tangent_pts[~vis] = np.nan
    elif method == TangentPoints.RFTA:
        # Reach for the Arcs: use the reconstructed point cloud as tangent points.
        # NOTE: this point cloud is NOT 1:1 with the input samples (one sphere can
        # contribute several or zero points), so only the useRBF=True reconstruction
        # path (zero-level constraints) is valid for it.
        _, _, P, _ = gpy.reach_for_the_arcs(
            points, distances, return_point_cloud=True,
            screening_weight=screening_weight, parallel=True, force_cpu=False)
        tangent_pts = np.asarray(P, dtype=np.float64)
    elif method == TangentPoints.MES:
        # Maximal Empty Spheres: use the reconstructed oriented point cloud as tangent points.
        # MES does NOT emit one contact point per input sample (fully-covered / interior
        # samples produce none), so the count differs from len(points) and there is no 1:1
        # correspondence -- again only the useRBF=True path is valid for it.
        MESReconstruction = _import_mes()
        *_, P, _ = MESReconstruction(points, distances, screening_weight=screening_weight, return_oriented_points=True)
        tangent_pts = np.asarray(P, dtype=np.float64)
        if tangent_pts.size == 0:
            raise RuntimeError("MES returned no contact points; cannot build a tangent-point set.")
    elif method == TangentPoints.GT:
        # Ground truth: the tangent point of each SDF sample is its closest point on the GT
        # mesh (the exact projection onto the true surface), 1:1 with points -- valid for
        # both reconstruction paths.
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int32)
        _, _, closest = igl.point_mesh_squared_distance(points, V, F)
        tangent_pts = np.asarray(closest, dtype=np.float64)
    else:
        raise ValueError(f"Unknown method: {method}")
    n_valid = int(np.isfinite(tangent_pts).all(axis=1).sum())
    print(f"  [{method.value}] tangent points: {n_valid} valid / {len(tangent_pts)} total  (from {len(points)} SDF samples)")
    return tangent_pts, points, distances

def construct_mesh(tangent_pts, points, distances, useRBF : bool, options : Options, screening_weight=10):
    """
        If useRBF is True, construct mesh using RBF interpolation, otherwise use sPSR on the input points.
        tangent_pts: (N, 3) array of tangent points corresponding to the input points. These
            can be used to compute normals for the sPSR method, or treated as 0 value constraints for RBF interpolation.
        points: (N, 3) array of point coordinates.
        distances: (N,) array of signed distance values corresponding to the input points.
        options: the RBF hyperparameters (reg, interp_overlap, interp_partition, pair_local)
            and grid_len (for the adaptive extraction resolution) are read from here so this
            reconstruction mirrors test_our_method's RBF -- in particular reg must be passed through
            (the C++ ctor defaults to reg=1e-5, but Options.reg defaults to 0).
        screening_weight: PSR screening weight for the sPSR (useRBF=False) path.
        Returns a trimesh.Trimesh of the reconstructed surface.
    """
    # Invalid tangent points are flagged NaN by get_tangent_points (non-visible projections);
    # drop them before reconstruction.
    valid = np.isfinite(tangent_pts).all(axis=1)
    if useRBF:
        # Fit the C++ RBF (PU) interpolator to the SDF samples (value constraints) plus the
        # valid tangent points (zero-level constraints), then extract the surface with the
        # C++ dual-contouring path. Plain value RBF -- no gradient/Hermite constraints; the
        # tangent points carry the surface information. No 1:1 correspondence between
        # tangent_pts and points is required here.
        sdf_cpp = _import_sdf_cpp()
        resolution = _adaptive_resolution(points, options.grid_len)
        print(f"Grid resolution for surface extraction: {resolution}")
        tp = tangent_pts[valid]
        interp = sdf_cpp.PUInterpolator(kernel='cubic', overlap=options.interp_overlap, reg=options.reg,
                                        partition=options.interp_partition, pair_local=options.pair_local,
                                        verbose=False)
        fit_pts = np.vstack([points, tp])
        fit_vals = np.concatenate([distances, np.zeros(len(tp))])
        interp.fit(fit_pts, fit_vals)
        bbox_min = points.min(axis=0) - options.extract_padding
        bbox_max = points.max(axis=0) + options.extract_padding
        verts, faces = interp.extract_surface(
            bbox_min=bbox_min, bbox_max=bbox_max,
            nx=resolution, ny=resolution, nz=resolution, iso=0.0, chunk_size=5000,
            lipschitz_postfix=False, use_dual_contouring=True)
        return trimesh.Trimesh(vertices=verts, faces=faces)
    else:
        # Screened Poisson on the tangent points. Normals are derived from the SDF samples:
        # (point - tangent) points away from the surface for outside samples; flipping by the
        # sign of the distance orients every normal outward. Requires tangent_pts 1:1 with
        # points (true for OURS); restrict to the valid (index-aligned) subset.
        tp = tangent_pts[valid]
        pts = points[valid]
        dst = distances[valid]
        d = pts - tp
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n < 1e-12] = 1.0
        normals = (d / n) * np.sign(dst)[:, np.newaxis]
        Vr, Fr = gpy.point_cloud_to_mesh(tp, normals, method='PSR', psr_screening_weight=screening_weight)
        recon = trimesh.Trimesh(vertices=Vr, faces=Fr)
        # sPSR can emit spurious closed "bubble" components floating outside the sampled
        # region; drop any component lying entirely outside the input (SDF sample) bbox.
        bmin, bmax = points.min(axis=0), points.max(axis=0)
        comps = recon.split(only_watertight=False)
        if len(comps) > 1:
            kept = [c for c in comps
                    if np.any(np.all((np.asarray(c.vertices) >= bmin)
                                     & (np.asarray(c.vertices) <= bmax), axis=1))]
            if not kept:  # never emit an empty mesh
                kept = [max(comps, key=lambda m: len(m.faces))]
            recon = trimesh.util.concatenate(kept)
        return recon

if __name__ == "__main__":
    t0 = time.perf_counter()
    seed = 1
    data_dir = 'examples'
    for length in [30]:
        options = Options(name='bunny', grid_len=length, verbose=True)
        points, distances = test_our_method(options, save_gtmesh=False)
        # test_rfta(options, screening_weight=10, parallel=True, sdf=(points, distances))
        # test_mc(options, save_gtmesh=False, sdf=(points, distances))
        # test_mes(options, save_gtmesh=False, screening_weight=10, sdf=(points, distances))
        # Tangent points from one method, surface rebuilt by RBF (useRBF=True) or sPSR (False):
        # tangent_pts, points, distances = get_tangent_points(options, TangentPoints.GT)
        # recon = construct_mesh(tangent_pts, points, distances, useRBF=True, options=options)
        # recon.export(f'out/{options.name}/tangent_gt_rbf_{length}.obj')

    elapsed = time.perf_counter() - t0
    print(f"  ⏱  {'Total execution time':<30} {elapsed:>7.2f} s")
