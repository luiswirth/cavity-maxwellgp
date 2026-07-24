import numpy as np


def green_scalar(r, k):
    return np.exp(1j * k * r) / (4 * np.pi * r)


def green_dyadic(rv, k):
    r = np.linalg.norm(rv)
    rhat = rv / r
    phi = green_scalar(r, k)
    transverse = k**2 + 1j * k / r - 1 / r**2
    radial = -(k**2) - 3j * k / r + 3 / r**2
    return (1j / k) * phi * (transverse * np.eye(3) + radial * np.outer(rhat, rhat))


def incident_field(x, z, k, p):
    return green_dyadic(x - z, k) @ p


def incident_field_batch(X, z, k, p):
    rv = X - z
    r = np.linalg.norm(rv, axis=1)
    rhat = rv / r[:, None]
    phi = np.exp(1j * k * r) / (4 * np.pi * r)
    transverse = k**2 + 1j * k / r - 1 / r**2
    radial = -(k**2) - 3j * k / r + 3 / r**2
    rhat_p = rhat @ p
    return (1j / k) * phi[:, None] * (
        transverse[:, None] * p + radial[:, None] * rhat_p[:, None] * rhat
    )
