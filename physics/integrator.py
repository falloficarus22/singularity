import taichi as ti

from .schwarzschild import geodesic_rhs


@ti.func
def rk4_step(state, rs, e_cons, dl):
    k1 = geodesic_rhs(state, rs, e_cons)
    k2 = geodesic_rhs(state + 0.5 * dl * k1, rs, e_cons)
    k3 = geodesic_rhs(state + 0.5 * dl * k2, rs, e_cons)
    k4 = geodesic_rhs(state + dl * k3, rs, e_cons)

    return state + (dl / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
