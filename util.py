import numpy as np
import trimesh


def mesh_distances(recon : trimesh.Trimesh, gt_mesh : trimesh.Trimesh, verbose=False,
                   n_samples: int = 100_000, f1_tau: float = 0.01):
    """
    Surface-sampled Hausdorff / Chamfer / F1 between two meshes.

    Both meshes are uniformly sampled on their surface (n_samples points each);
    distances are computed symmetrically between each sample set and the *other*
    mesh's surface via an AABB tree. This is fair regardless of tessellation
    density (vertex-only metrics penalise sparsely-sampled reconstructions
    unfairly even when the surface is correct).

    Hausdorff is the symmetric max (intentionally outlier-sensitive), computed
    from mesh vertices (not samples) so it is deterministic — for triangulated
    surfaces in generic position the argmax lies on a vertex of one mesh.
    Chamfer is the L1 symmetric mean of the two directional means.

    F1 threshold ``f1_tau`` is interpreted as a fraction of the GT mesh's
    bounding-box diagonal (Occupancy Networks / DeepSDF convention). With
    f1_tau=0.01 a point counts as a "hit" if it lies within 1% of the GT
    diagonal of the other surface. This keeps the threshold geometrically
    meaningful across meshes of different aspect ratios (thin slabs vs cubes).
    """
    import igl
    recon_pts, _ = trimesh.sample.sample_surface(recon, n_samples)
    gt_pts,    _ = trimesh.sample.sample_surface(gt_mesh, n_samples)

    gt_V = np.asarray(gt_mesh.vertices, dtype=np.float64)
    gt_F = np.asarray(gt_mesh.faces, dtype=np.int32)
    rc_V = np.asarray(recon.vertices, dtype=np.float64)
    rc_F = np.asarray(recon.faces, dtype=np.int32)

    sqrD_r2g, _, _ = igl.point_mesh_squared_distance(
        np.asarray(recon_pts, dtype=np.float64), gt_V, gt_F)
    sqrD_g2r, _, _ = igl.point_mesh_squared_distance(
        np.asarray(gt_pts, dtype=np.float64), rc_V, rc_F)
    d_r2g = np.sqrt(sqrD_r2g)
    d_g2r = np.sqrt(sqrD_g2r)

    sqrD_rV2g, _, _ = igl.point_mesh_squared_distance(rc_V, gt_V, gt_F)
    sqrD_gV2r, _, _ = igl.point_mesh_squared_distance(gt_V, rc_V, rc_F)
    hausdorff = float(np.sqrt(max(sqrD_rV2g.max(), sqrD_gV2r.max())))
    chamfer = (d_r2g.mean() + d_g2r.mean()) / 2

    # Scale f1_tau by GT bbox diagonal so the threshold is geometrically
    # consistent across mesh aspect ratios. f1_tau=0.01 → 1% of GT diagonal.
    gt_extents = gt_V.max(axis=0) - gt_V.min(axis=0)
    gt_diag = float(np.linalg.norm(gt_extents))
    tau_abs = f1_tau * gt_diag
    precision = float((d_r2g < tau_abs).mean())
    recall    = float((d_g2r < tau_abs).mean())
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if verbose:
        print(f"  Hausdorff: {hausdorff:.6f}  Chamfer: {chamfer:.6f}  "
              f"F1@{f1_tau}*diag={tau_abs:.4f}: {f1:.4f} "
              f"(P={precision:.4f} R={recall:.4f})")
    return hausdorff, chamfer, f1

