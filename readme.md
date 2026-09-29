# RBF Your SDF: Radial Basis Function Interpolation of Signed Distance Fields with Implied Tangent Points

This repository serves as the implementation of the paper.

**Project Page:** <https://cragl.cs.gmu.edu/rbfyoursdf/>

## Installation

```bash
pip install "rbfyoursdf @ git+https://github.com/CocoYard/RBF-your-SDF"
```

## Usage

```python
import rbfyoursdf

# points: (N, 3) sample positions, values: (N,) signed distances (negative inside)
result = rbfyoursdf.main_algorithm(points, values, max_iters=10)
verts, faces = result.interpolator.extract_surface(
    bbox_min=points.min(axis=0), bbox_max=points.max(axis=0),
    nx=128, ny=128, nz=128, use_dual_contouring=True)
```

`result.projections` holds each sample's tangent point and `result.visibility_mask`
whether it was used. See `help(rbfyoursdf.main_algorithm)` for all keyword arguments.

## Contents

- `rbfyoursdf/` — the Python package: `main_algorithm` and the C++ bindings.
- `cpp/` — the C++ core (gradient estimation, tangent-point projection, partition-of-unity
  RBF fitting, surface extraction), exposed to Python as the `rbfyoursdf.rbfyoursdf_cpp` module.
- `demo/SDF_to_surface_3D.py` — samples an SDF from a mesh, runs our method, and reconstructs
  the surface; also runs the baselines below.
- `demo/additional_experiments/` — scripts that reproduce the additional experiments.
- `examples/` — the test meshes.

## Building

To run the demo and our experiments, clone the repository and run locally.

```bash
uv sync
```

or

```bash
pip install . --group demo
```

## Running

```bash
uv run demo/SDF_to_surface_3D.py
```

reconstructs `examples/bunny.obj` from a 30³ SDF grid with our method and writes the
mesh to `out/`. Edit `main()` to pick another model, resolution or method.

## Methods

- **Ours** — `test_our_method`
- **Reach for the Arcs** — `test_rfta` (optional, via `gpytoolbox`)
- **Maximal Empty Spheres** — `test_mes` (optional, needs a separate build; see
  [demo/additional_experiments/README.md](demo/additional_experiments/README.md))
- **Marching cubes** on the sample grid — `test_mc`

**Is our improved performance due to better tangent points or RBF interpolation?** 
`get_tangent_points` takes the surface points implied by one method (ground truth, ours, 
RFTA or MES) and `construct_mesh` rebuilds the surface from them with either our RBF or 
screened Poisson, isolating the quality of the tangent points from the reconstruction step.

## Experiments

Each script in `demo/additional_experiments/` reproduces one table; see its
[README](demo/additional_experiments/README.md) for details.

| script | what it tests |
| --- | --- |
| `scattered.py` | samples at random positions instead of a grid |
| `truncation.py` | samples only in a narrow band around the surface |
| `noise.py` | noisy distance values |
| `convergence.py` | reconstruction quality vs. number of iterations |
| `degen_tol.py` | sensitivity to the short-arc threshold |
