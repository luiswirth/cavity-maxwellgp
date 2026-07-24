import argparse
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from maxwellgp import GaussianProcess, MaxwellKernel
from maxwellgp.utils import fibonacci_sphere

from .analytic import incident_field_batch

jax.config.update("jax_enable_x64", True)

JITTER = 1e-8


@dataclass
class GPConfig:
    n_boundary: int = 1200
    log_noise: float = -12.0
    opt_noise: bool = False
    opt_steps: int = 200

    @classmethod
    def from_args(cls, args):
        return cls(args.n_boundary, args.log_noise, args.opt_noise, args.opt_steps)


def load_config(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    k, a, b, c, _n = lines[0].split()
    n = int(_n)
    semiaxes = np.array([float(a), float(b), float(c)])
    data = np.array([[float(v) for v in ln.split()] for ln in lines[1:]])
    assert len(data) == n, f"config declares N={n} but has {len(data)} rows"
    return float(k), semiaxes, data[:, 0:3], data[:, 3:6], data[:, 6:9]


def boundary_collocation(semiaxes, n):
    u = np.asarray(fibonacci_sphere(n))
    points = u * semiaxes
    normals = points / semiaxes**2
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def tangential_trace(Ei, normals):
    En = np.sum(Ei * normals, axis=1, keepdims=True)
    return -(Ei - En * normals)


def optimize_log_noise(kernel, log_noise0, X_train, Y, steps, lr=0.05):
    ln = jnp.asarray(log_noise0)
    opt = optax.adam(lr)
    state = opt.init(ln)

    def loss_of(ln):
        return GaussianProcess(kernel, log_noise=ln).nlml(X_train, Y)

    @jax.jit
    def step(ln, state):
        loss, g = jax.value_and_grad(loss_of)(ln)
        updates, state = opt.update(g, state)
        ln = jnp.clip(optax.apply_updates(ln, updates), -12.0, 0.0)
        return ln, state, loss

    loss = loss_of(ln)
    for _ in range(steps):
        ln, state, loss = step(ln, state)
    return float(ln), float(loss)


def fit(cfg, semiaxes, k, Y, n_spectral):
    bnd_points, bnd_normals = boundary_collocation(semiaxes, cfg.n_boundary)
    X_train = jnp.asarray(np.concatenate([bnd_points, bnd_normals], axis=1))

    kernel = MaxwellKernel(n_spectral=n_spectral, wavenumber=k, trace="tangential")
    log_noise = cfg.log_noise
    if cfg.opt_noise:
        log_noise, _ = optimize_log_noise(kernel, log_noise, X_train, Y, cfg.opt_steps)
        print(f"tuned log_noise = {log_noise:.4f} (eps={np.exp(log_noise):.3e})")
    model = GaussianProcess(kernel, log_noise=log_noise)
    return model, model.condition(X_train, Y, jitter=JITTER)


def _nlml(post, model, Y):
    # Marginal likelihood reusing the conditioned factor (no re-factorization):
    # Phi_Y = A mu_w = L (L^H mu_w), and the nlml fit term is <Phi_Y, mu_w>.
    L, w = post.L, post.mu_w
    M, J = Y.shape
    Phi_Y = L @ (L.conj().T @ w)
    data_fit = 0.5 * (jnp.vdot(Y, Y).real / jnp.exp(model.log_noise)
                      - jnp.sum((Phi_Y.conj() * w).real))
    logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L).real))
    logdet_C = logdet_A + M * model.log_noise - jnp.sum(model.kernel.log_weights)
    return float(data_fit + J * (0.5 * logdet_C + 0.5 * M * jnp.log(2.0 * jnp.pi)))


def assemble_operator(cfg, semiaxes, k, points, e1, e2, n_spectral):
    configs = []
    for i in range(len(points)):
        n = points[i] / np.linalg.norm(points[i])
        configs.append((points[i], n, e1[i]))
        configs.append((points[i], n, e2[i]))
    n_cfg = len(configs)

    bnd_points, bnd_normals = boundary_collocation(semiaxes, cfg.n_boundary)
    cols = [tangential_trace(incident_field_batch(bnd_points, z, k, p), bnd_normals).reshape(-1)
            for z, _, p in configs]
    Y = jnp.asarray(np.stack(cols, axis=1))

    model, post = fit(cfg, semiaxes, k, Y, n_spectral)

    X_query = jnp.asarray(np.stack([np.concatenate([x, nrm]) for x, nrm, _ in configs]))
    Q = jnp.asarray(np.stack([q for _, _, q in configs]))
    Phi_q = model.kernel.features(X_query).reshape(-1, n_cfg, 3)
    Psi = jnp.einsum("fic,ic->fi", Phi_q, Q)
    T = np.asarray(post.mean(Psi))
    Sigma = np.asarray(post.cov(Psi))
    nlml = _nlml(post, model, Y)
    return T, Sigma, nlml, post, model


def run_operator(args):
    # secs and mem are captured by /usr/bin/time in the run harness, not here.
    ns, nb = args.n_spectral, args.n_boundary
    k, semiaxes, points, e1, e2 = load_config(args.config)
    cfg = GPConfig.from_args(args)
    T, Sigma, nlml, post, model = assemble_operator(cfg, semiaxes, k, points, e1, e2, ns)

    cond = float(np.linalg.cond(np.asarray(post.L @ post.L.conj().T)))
    recip = np.linalg.norm(T - T.T) / np.linalg.norm(T)
    std = np.sqrt(np.clip(np.real(np.diag(Sigma)), 0.0, None))
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"T_ns{ns}_nb{nb}.npy")
    np.save(out, T)
    np.save(os.path.join(args.outdir, f"Sigma_ns{ns}_nb{nb}.npy"), Sigma)
    print(f"dofs={2 * ns}")
    print(f"cond={cond:.6e}")
    print(f"recip={recip:.3e}")
    print(f"mean_std={std.mean():.6e}")
    print(f"log_noise={float(model.log_noise):.6f}")
    print(f"nlml={nlml:.6e}")
    print(f"wrote {out}")


def run_field(args):
    k, semiaxes, *_ = load_config(args.config)
    a, b, c = semiaxes
    z = np.array(args.source, dtype=float)
    p = np.array(args.pol, dtype=float)

    cfg = GPConfig.from_args(args)
    bnd_points, bnd_normals = boundary_collocation(semiaxes, cfg.n_boundary)
    y = jnp.asarray(tangential_trace(incident_field_batch(bnd_points, z, k, p),
                                     bnd_normals).reshape(-1, 1))

    model, post = fit(cfg, semiaxes, k, y, args.n_spectral)

    xs = np.linspace(-1.05 * a, 1.05 * a, args.ngrid)
    zs = np.linspace(-1.05 * c, 1.05 * c, args.ngrid)
    XX, ZZ = np.meshgrid(xs, zs)
    pts = np.stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()], axis=1)

    mean_chunks, var_chunks = [], []
    for i in range(0, len(pts), args.batch):
        phi = model.kernel.feature_map.full(jnp.asarray(pts[i : i + args.batch]))
        mean_chunks.append(np.asarray(post.mean(phi)))
        var_chunks.append(np.asarray(post.var(phi)))
    field6 = np.concatenate(mean_chunks).reshape(-1, 6)
    var6 = np.concatenate(var_chunks).reshape(-1, 6)
    Escat = field6[:, :3]
    Escat_std = np.sqrt(var6[:, :3])
    Einc = incident_field_batch(pts, z, k, p)
    Etot = Einc + Escat

    inside = (pts[:, 0] ** 2 / a**2 + pts[:, 1] ** 2 / b**2 + pts[:, 2] ** 2 / c**2) <= 1.0
    ng = args.ngrid
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        xs=xs, zs=zs,
        Escat=Escat.reshape(ng, ng, 3),
        Escat_std=Escat_std.reshape(ng, ng, 3),
        Einc=Einc.reshape(ng, ng, 3),
        Etot=Etot.reshape(ng, ng, 3),
        mask=inside.reshape(ng, ng),
        semiaxes=semiaxes, source=z, pol=p, k=k,
    )
    print(f"wrote {args.out}  (slice {ng}x{ng}, source={z.tolist()}, pol={p.tolist()})")


def _interior_points(semiaxes, n_per_shell, shells=(0.4, 0.7)):
    pts = [np.asarray(fibonacci_sphere(n_per_shell)) * semiaxes * s for s in shells]
    return np.concatenate(pts, axis=0)


def run_ksweep(args):
    """Betcke-Trefethen subspace angle method for PEC cavity resonance detection.

    At each k, orthonormalize [boundary_trace; interior_field] and compute
    sigma_min of the boundary block. This dips to ~0 at cavity resonances
    without confounding basis overcompleteness with physical resonances.
    """
    _, semiaxes, *_ = load_config(args.config)
    R = float(np.max(semiaxes))
    ks = np.linspace(args.kmin, args.kmax, args.nk)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "ksweep.csv")
    with open(out, "w") as f:
        f.write("k,sigma_min\n")
        for k in ks:
            n_spec = int(min(args.nmax, max(args.nmin, round(args.alpha * (k * R) ** 2))))
            n_b = int(max(250, round(args.oversample * 2 * n_spec / 3)))
            n_i = max(40, n_spec // 4)
            bnd_points, bnd_normals = boundary_collocation(semiaxes, n_b)
            Xb = jnp.asarray(np.concatenate([bnd_points, bnd_normals], axis=1))
            Xi = jnp.asarray(_interior_points(semiaxes, n_i))
            fm = MaxwellKernel(n_spectral=n_spec, wavenumber=float(k)).feature_map
            Ab = np.asarray(fm.tangential(Xb)).T   # (3*n_b, F)
            Ai = np.asarray(fm.full(Xi)).T          # (6*n_i, F)
            M = np.vstack([Ab, Ai])
            Q, _ = np.linalg.qr(M, mode="reduced")
            Qb = Q[: Ab.shape[0], :]
            s = float(np.linalg.svd(Qb, compute_uv=False).min())
            f.write(f"{k:.6f},{s:.6e}\n")
            print(f"k={k:.4f} n_spec={n_spec:>4} sigma_min={s:.6e}")
    print(f"wrote {out}")


def add_common(sp):
    sp.add_argument("--config", default="res/config_ellipse.txt")
    sp.add_argument("--n-spectral", type=int, default=256)
    sp.add_argument("--n-boundary", type=int, default=1200)
    sp.add_argument("--log-noise", type=float, default=-12.0)
    sp.add_argument("--opt-noise", action=argparse.BooleanOptionalAction, default=False)
    sp.add_argument("--opt-steps", type=int, default=200)


def main():
    ap = argparse.ArgumentParser(description="Maxwell-GP PEC ellipsoidal cavity")
    sub = ap.add_subparsers(dest="cmd", required=True)

    op = sub.add_parser("operator", help="assemble the dipole reaction operator T")
    add_common(op)
    op.add_argument("--outdir", default="out/ellipse")
    op.set_defaults(func=run_operator)

    fld = sub.add_parser("field", help="evaluate the field on a slice for one dipole")
    add_common(fld)
    fld.add_argument("--source", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    fld.add_argument("--pol", type=float, nargs=3, required=True, metavar=("PX", "PY", "PZ"))
    fld.add_argument("--ngrid", type=int, default=400)
    fld.add_argument("--batch", type=int, default=4000)
    fld.add_argument("--out", default="out/ellipse/field.npz")
    fld.set_defaults(func=run_field)

    ks = sub.add_parser("ksweep", help="sweep wavenumber, record minimum subspace angle")
    ks.add_argument("--config", default="res/config_ellipse.txt")
    ks.add_argument("--kmin", type=float, default=0.25)
    ks.add_argument("--kmax", type=float, default=4.0)
    ks.add_argument("--nk", type=int, default=200)
    ks.add_argument("--alpha", type=float, default=1.2,
                    help="n_spectral ~ alpha*(k*R)^2")
    ks.add_argument("--nmin", type=int, default=40)
    ks.add_argument("--nmax", type=int, default=600)
    ks.add_argument("--oversample", type=float, default=3.5,
                    help="n_boundary = oversample * 2*n_spectral/3")
    ks.add_argument("--outdir", default="out/ksweep/ellipse")
    ks.set_defaults(func=run_ksweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
