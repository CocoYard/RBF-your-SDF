import os
# torch (pulled in lazily via neural_sdf) and sdf_cpp each bundle their own
# libomp; without this the second one to initialize aborts (OMP Error #15). Set
# before anything can import torch so the neural-SDF source can run in-process.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import trimesh
import gpytoolbox as gpy
import igl
import numpy as np
import time

# Seed for all randomness in generate_test_mesh_data (scatter sampling, noise).
# __main__ overrides this; importers can set it via `sdf3d.seed = ...` like data_dir.
# None = nondeterministic.
seed = None

class Options:
    def __init__(self, grid_len=20, gt_mesh=None, clamp=True, max_iters=10, name='horse', lr=0.2, optim_steps=5,
                 turn_off_short_arcs=False, export_short_arcs=False, export_projections=False, reg=0,
                 use_gt_gradients=False, interpolator_type='PU', interp_partition='sphere', overlap=0.2, cpp_dc=True,
                 post_processing=False, iter_gradient_finding='optimize', verbose=True,
                pair_local=False, noise=0, bound=1, scatter=False, neural_sdf=None,
                grad_optimizer='bfgs', extract_padding=0.0, degen_tol=1e-5):
        self.grid_len = grid_len
        self.max_iters = max_iters
        self.clamp = clamp
        self.turn_off_short_arcs = turn_off_short_arcs
        self.name = name
        self.path_to_obj = f'{data_dir}/{name}.obj'
        self.export_short_arcs = export_short_arcs  # whether to export short arcs .glb for visualization
        self.export_projections = export_projections  # export gradients .glb for visualization
        self.use_gt_gradients = use_gt_gradients
        self.interpolator_type = interpolator_type  # 'Duchon' or 'PU'
        self.interp_partition = interp_partition  # 'box' or 'fps' or 'sphere', only for PU interpolator
        self.interp_overlap = overlap
        self.pair_local = pair_local  # PU: pair each local RBF solve with missing input/projection partners
        self.post_processing = post_processing
        self.reg = reg
        self.iter_gradient_finding = iter_gradient_finding  # 'optimize' or 'sample'
        self.cpp_dc = cpp_dc
        self.verbose = verbose
        self.lr = lr
        self.optim_steps = optim_steps  # quasi-Newton steps per outer iteration
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

        self.gt_gradients = None  # set it manually if you want to use GT gradients for testing, e.g. from the intermediate output of generate_test_mesh_data
        self.gt_mesh = gt_mesh  # set it manually if you want to compute distances to GT mesh at the end, e.g. from the intermediate output of generate_test_mesh_data

        self.tolerance = None  # set it manually if you want to adjust the tolerance for clamping, e.g. based on the mean spacing of the input points

        self.noise = noise
        self.bound = bound
        self.scatter = scatter

        # Source the SDF from a neural field trained on the obj, computed in-process
        # (no npz round-trip). None = exact mesh SDF; 'gt' / 'pc' / 'igr' = neural.
        self.neural_sdf = neural_sdf
        self.neural_retrain = False  # retrain the neural field instead of using the cached weights
    def print(self):
        print(f"Options: grid_len={self.grid_len}, name={self.name},"
              f" max_iters={self.max_iters}, clamp={self.clamp},"
              f" turn_off_short_arcs={self.turn_off_short_arcs},"
              f" export_short_arcs={self.export_short_arcs},"
              f" export_projections={self.export_projections},"
              f" use_gt_gradients={self.use_gt_gradients},"
              f" interpolator_type={self.interpolator_type},"
              f" interp_partition={self.interp_partition}",
              f" interp_overlap={self.interp_overlap}",
              f" pair_local={self.pair_local}",
              f" grad_optimizer={self.grad_optimizer}",
              f" degen_tol={self.degen_tol}",
              f" lr={self.lr}")

class Tolerance:
    def __init__(self, clamp_radius_ratio=0.2, clamp_sdf_tol=1e-3, angle_tol=np.radians(15)):
        # 0.2 means gradient rotates 11.5 degrees at most
        # 0.1 means gradient rotates 5.7 degrees at most
        self.clamp_radius_ratio = clamp_radius_ratio # for clamping to the nearest arc point
        self.clamp_sdf_tol = clamp_sdf_tol           # for clamping to the optimal point on visible boundary
        self.float_tol = 1e-8
        self.angle_tol = angle_tol

def generate_test_mesh_data( path_to_mesh, outbase, grid_len=10, save=False, noise=0.0, bound=1.0, scatter=False ):
    '''
    Loads a mesh from the given path and computes signed distances and gradients for its vertices.
    Parameters:
    path_to_mesh: str
        The file path to the mesh.
    Returns:
    points: (N, 3) array of vertex coordinates
        The vertices of the mesh.
    distances: (N,) array of signed distance values
        The signed distance values for each vertex.
    gradients: (N, 3) array of gradient vectors
        The gradient vectors at each vertex.
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
    sq_dists, face_ids, closest = igl.point_mesh_squared_distance(points, V, F)
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

    # Filter out points that are too close to the surface (within 0.1 units), also remove respective gradients
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
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh, noise=options.noise, bound=options.bound)  # Generate new data with 4096 points
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
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh, bound=options.bound)  # Generate new data with 4096 points
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
    gpy.write_mesh(f"{out_dir}/" + fname, *R_cgal)

    print(f"Exported: {out_dir}/" + fname)

def _build_cpp_options(options : Options):
    """ Translate a Python Options into a C++ sdf_cpp.Options and return it. """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp', 'build'))
    import sdf_cpp
    cpp_opts = sdf_cpp.Options()
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
    cpp_opts.grad_optimizer = getattr(options, 'grad_optimizer', 'bfgs')
    cpp_opts.lr = options.lr
    cpp_opts.optim_steps = options.optim_steps
    cpp_opts.degen_tol = options.degen_tol
    cpp_opts.verbose = options.verbose
    if options.use_gt_gradients:
        cpp_opts.gt_gradients = options.gt_gradients
    if options.tolerance is not None:
        cpp_opts.tolerance.clamp_radius_ratio = options.tolerance.clamp_radius_ratio
        cpp_opts.tolerance.clamp_sdf_tol = options.tolerance.clamp_sdf_tol
        cpp_opts.tolerance.angle_tol = options.tolerance.angle_tol
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
    if options.neural_sdf is not None:
        # Source the SDF from a neural field trained on the obj, in-process — no
        # npz round-trip. torch and sdf_cpp coexist thanks to KMP_DUPLICATE_LIB_OK
        # (set at module import). options.neural_sdf selects the mode:
        #   'gt' / 'pc' / 'igr'  (see neural_sdf.train_neural_sdf).
        # No artificial noise is injected: the neural field is itself the
        # imperfect (learned, smoothed) SDF, which is the point of the test. The
        # exact mesh is kept for error evaluation.
        from neural_sdf import generate_neural_sdf_data
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, _ = generate_neural_sdf_data(
            path_to_obj, base_name, grid_len=grid_len, mode=options.neural_sdf,
            bound=options.bound, scatter=options.scatter,
            retrain=options.neural_retrain, verbose=options.verbose)
        options.gt_mesh = mesh  # exact mesh kept for error evaluation
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh, noise=options.noise, bound=options.bound, scatter=options.scatter)  # Generate new data with 4096 points
        options.gt_gradients = gt_gradients
        options.gt_mesh = mesh

    # Create and fit the interpolator
    timer = time.perf_counter()
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cpp', 'build'))
    import sdf_cpp
    cpp_opts = _build_cpp_options(options)
    result = sdf_cpp.main_algorithm(points, distances, cpp_opts)
    _vis = np.asarray(result.visibility_mask).ravel()
    print(f"Final visibility: {int((_vis != 0).sum())}/{len(_vis)} ({100.0 * (_vis != 0).mean():.2f}%)")
    print(f"  ⏱  {'Interpolator fitted':<30} {time.perf_counter() - timer:>7.2f} s")

    """ ========================= output post+dc ========================= """
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
    post_str = 'post' if options.post_processing else ''
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

def check_mesh_error(dir_to_meshes, path_to_gt, edge_chamfer=False):
    """ Compute the mesh distance (Hausdorff and Chamfer) between meshes in dir_to_meshes and the ground truth mesh at path_to_gt. """
    gt_mesh = trimesh.load(path_to_gt, force='mesh')
    # Normalize the mesh to fit within a unit cube
    min = np.min( gt_mesh.vertices, axis=0 )
    max = np.max( gt_mesh.vertices, axis=0 )
    gt_mesh.vertices -= (min + max) / 2
    gt_mesh.vertices /= np.max( max - min )

    meshes = os.listdir(dir_to_meshes)
    meshes.sort()
    print(f"{os.path.basename(dir_to_meshes):<30}")
    import edgeChamfer
    for mesh_file in meshes:
        if mesh_file.endswith('.obj'):
            mesh = trimesh.load(os.path.join(dir_to_meshes, mesh_file), force='mesh')
            haus, chamfer, f1 = mesh_distances(mesh, gt_mesh)
            if edge_chamfer:
                ecd, ef1 = edgeChamfer.compute_ecd(mesh, gt_mesh, sample_num=1000_000)
            print(f"{mesh_file:<50} against ground truth...", end='')
            if edge_chamfer:
                print(f"  Hausdorff: {haus:.5f}  Chamfer: {chamfer:.7f}  F1: {f1:.4f}  EdgeChamfer: {ecd:.5f}  EdgeF1: {ef1:.4f}")
            else:
                print(f"  Hausdorff: {haus:.5f}  Chamfer: {chamfer:.7f}  F1: {f1:.4f}")

def test_mc(options : Options, save_gtmesh=False, sdf=None):
    """ Marching cubes directly on the (grid) SDF samples, no interpolation; export to out/<name>/. """
    grid_len = options.grid_len
    path_to_obj = options.path_to_obj
    options.print()
    if sdf is not None:
        points, distances = sdf
    else:
        base_name = path_to_obj.split('/')[-1].split('.')[0]
        mesh, points, distances, gt_gradients = generate_test_mesh_data(path_to_obj, base_name, grid_len=grid_len, save=save_gtmesh)  # Generate new data with 4096 points
        options.gt_gradients = gt_gradients
        options.gt_mesh = mesh
    # --- Second window: marching cubes directly on sample points (原始网格点) ---
    # 从点坐标反推网格结构，无需插值
    xs = np.unique(np.round(points[:, 0], 8))
    ys = np.unique(np.round(points[:, 1], 8))
    zs = np.unique(np.round(points[:, 2], 8))
    nx, ny, nz = len(xs), len(ys), len(zs)
    ix = np.searchsorted(xs, np.round(points[:, 0], 8))
    iy = np.searchsorted(ys, np.round(points[:, 1], 8))
    iz = np.searchsorted(zs, np.round(points[:, 2], 8))
    grid_values_direct = np.ones((nx, ny, nz))  # 缺失点默认为外部(+1)
    grid_values_direct[ix, iy, iz] = distances
    sp = ((xs[-1]-xs[0])/(nx-1), (ys[-1]-ys[0])/(ny-1), (zs[-1]-zs[0])/(nz-1))
    from skimage.measure import marching_cubes
    verts2, faces2, _, _ = marching_cubes(grid_values_direct, level=0.0, spacing=sp)
    verts2 += np.array([xs[0], ys[0], zs[0]])
    out_dir = 'out/' + path_to_obj.split('/')[-1].split('.')[0]
    os.makedirs(out_dir, exist_ok=True)
    trimesh.Trimesh(vertices=verts2, faces=faces2).export(f'{out_dir}/sample_points_{grid_len}.obj')
    print(f"Exported: {out_dir}/sample_points_{grid_len}.obj")

if __name__ == "__main__":
    t0 = time.perf_counter()
    seed = 1
    batch = False
    data_dir = 'examples'
    if batch:
        for name in ['loewe']:
            for grid_len in [10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]:  # 20^3=8000 points, 30^3=27000 points
                options = Options(name=name, grid_len=grid_len, max_iters=13, clamp=False, cpp_dc=True, verbose=True,
                        export_short_arcs=False, export_projections=False, turn_off_short_arcs=True,
                        use_gt_gradients=False, interpolator_type='PU', interp_partition='sphere',
                        overlap=0.2, reg=0, post_processing=False, iter_gradient_finding='optimize')
                # test_our_method(options, save_gtmesh=False)
                # test_rfta(options, screening_weight=10, parallel=True)
                # test_mes(options, save_gtmesh=False, screening_weight=10)
            check_mesh_error(f'out/{name}', f'{data_dir}/{name}.obj')
    else:
        for length in [30]:
            options = Options(name='eiffel', grid_len=length, clamp=True, optim_steps=5)
            # options.grad_optimizer = "lbfgspp"  # 每点独立跑 LBFGS++
            # options.grad_optimizer = "ascent"    # 固定步长投影梯度上升(原方法)
            points, distances = test_our_method(options, save_gtmesh=False)
            # test_rfta(options, screening_weight=10, parallel=True, sdf=(points, distances))
            # test_mc(options, save_gtmesh=False, sdf=(points, distances))
            # test_mes(options, save_gtmesh=False, screening_weight=10, sdf=(points, distances))
        # check_mesh_error(f'out/{options.name}', f'{data_dir}/{options.name}.obj', edge_chamfer=True)

    elapsed = time.perf_counter() - t0
    print(f"  ⏱  {'Total execution time':<30} {elapsed:>7.2f} s")
