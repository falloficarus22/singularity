import taichi as ti


@ti.func
def get_f(r, rs):
    return 1.0 - rs / r


@ti.func
def geodesic_rhs(state, rs, e_cons):
    r, theta, phi, dr, dtheta, dphi = (
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
    )

    st = ti.sin(theta)
    ct = ti.cos(theta)

    f = get_f(r, rs)
    dt_dl = e_cons / f

    # Schwarzschild Geodesic Equations (Spheraical Coordinates)
    ddr = (
        -(rs / (2.0 * r**2)) * f * dt_dl**2
        + (rs / (2.0 * r**2 * f)) * dr**2
        + r * (dtheta**2 + st**2 * dphi**2)
    )

    ddtheta = -2.0 * dr * dtheta / r + st * ct * dphi**2
    ddphi = -2.0 * dr * dphi / r - 2.0 * (ct / st) * dtheta * dphi

    return ti.Vector([dr, dtheta, dphi, ddr, ddtheta, ddphi])
