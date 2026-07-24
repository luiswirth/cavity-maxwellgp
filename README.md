# cavity-maxwellgp

Ehrenpreis-Palamodov Gaussian process (EPGP) solver for the interior PEC cavity
reaction operator. A thin cavity-specific layer over
[maxwellgp](https://github.com/luiswirth/maxwellgp): it owns the analytic
dipole / incident-field physics (the dyadic Green's function) and assembles the
dipole transmit-to-receive reaction operator `T`, its posterior covariance, and
field slices. The plane-wave prior satisfies Maxwell's equations intrinsically.

Convergence is controlled by two parameters: `N_s` (`--n-spectral`, number of
plane-wave features) and `N_b` (`--n-boundary`, number of boundary conditioning
points). The cavity geometry is set by the semi-axes in `res/config_{shape}.txt`.

## Requirements

- [uv](https://docs.astral.sh/uv/) (handles Python and all dependencies)

`uv` resolves `maxwellgp` from GitHub over https; no SSH key needed. For local
development against a sibling `../maxwellgp` checkout, see the override note in
`pyproject.toml`.

## Run locally

`run_local.sh` runs the grid (or field slice / noise sweep) serially, no scheduler:

```bash
./run_local.sh grid            # 2D (N_s, N_b) grid, both shapes
./run_local.sh field ellipse   # field slice at the highest-fidelity grid config
./run_local.sh noise sphere    # fixed resolution, sweep assumed boundary noise
```

The taskfiles `euler/{grid,field,noise}.txt` define the runs. To run a single
operator or field slice directly:

```bash
uv run epgp-operator operator --config res/config_ellipse.txt --n-spectral 256 --n-boundary 4096 --outdir out/grid/ellipse
uv run epgp-operator field    --config res/config_ellipse.txt --n-spectral 1024 --n-boundary 8192 --source 0 0 1 --pol 1 0 0 --out out/field/ellipse/field.npz
```

## Run on a cluster (ETH Euler)

```bash
euler/submit_grid.sh  [shape]  # sbatch 2D (N_s, N_b) convergence grid
euler/submit_field.sh [shape]  # sbatch field slice at the highest-fidelity grid config
euler/submit_noise.sh [shape]  # sbatch assumed-noise sweep
euler/submit_ksweep.sh [shape] # sbatch wavenumber sweep (conditioning vs k)
```

The SLURM account and node constraint in `euler/run.sbatch` are Euler-specific
and flagged at the top of that file; adjust them for another SLURM site.

## Output

Per shape, under `out/{grid,field,noise}/{shape}/`:

- `T_ns{ns}_nb{nb}.npy` reaction operator, `Sigma_ns{ns}_nb{nb}.npy` its covariance
- `field.npz` field slice (field runs)
- `manifest.csv` columns `n_spectral,n_boundary,dofs,secs,mem_kb,cond`
- `provenance.csv` git commit, host, parameters, timestamp, log-noise, nlml

Collect into the benchmark harness with `cavity-benchmark/pull-euler.sh`.
