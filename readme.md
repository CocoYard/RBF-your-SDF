# RBF Your SDF: Radial Basis Function Interpolation of Signed Distance Fields with Implied Tangent Points

This repository serves as the implementation of the paper.

**Project Page:** https://cragl.cs.gmu.edu/rbfyoursdf/

## Contents

- `cpp/` — the C++ core (gradient estimation, tangent-point projection, partition-of-unity
  RBF fitting, surface extraction), exposed to Python as the `sdf_cpp` module.
- `SDF_to_surface_3D.py` — samples an SDF from a mesh, runs our method, and reconstructs
  the surface; also runs the baselines below.
- `additional_experiments/` — scripts that reproduce the additional experiments.
- `examples/` — the test meshes.

## Building

```bash
cd cpp
cmake -B build
cmake --build build -j
```

Missing C++ dependencies are fetched automatically. Build with the Python that you will run
the scripts with (e.g. an activated conda environment). Install the Python packages with

```bash
pip install -r requirements.txt
```

## Running

```bash
python SDF_to_surface_3D.py
```

reconstructs `examples/eiffel.obj` from a 30³ SDF grid with our method and writes the
mesh to `out/`. Edit the `__main__` block to pick another model, resolution or method.

## Methods

- **Ours** — `test_our_method`
- **Reach for the Arcs** — `test_rfta` (optional, via `gpytoolbox`)
- **Maximal Empty Spheres** — `test_mes` (optional, needs a separate build; see
  [additional_experiments/README.md](additional_experiments/README.md))
- **Marching cubes** on the sample grid — `test_mc`

**Is our improved performance due to better tangent points or RBF interpolation?** 
`get_tangent_points` takes the surface points implied by one method (ground truth, ours, 
RFTA or MES) and `construct_mesh` rebuilds the surface from them with either our RBF or 
screened Poisson, isolating the quality of the tangent points from the reconstruction step.

## Experiments

Each script in `additional_experiments/` reproduces one table; see its
[README](additional_experiments/README.md) for details.

| script | what it tests |
| --- | --- |
| `scattered.py` | samples at random positions instead of a grid |
| `truncation.py` | samples only in a narrow band around the surface |
| `noise.py` | noisy distance values |
| `convergence.py` | reconstruction quality vs. number of iterations |
| `degen_tol.py` | sensitivity to the short-arc threshold |
