import taichi as ti


@ti.func
def init_ray(pos, direction, rs):
    r = pos.norm()
    theta = ti.acos(pos.z / r)
    phi = ti.atan2(pos.y, pos.x)

    d = direction.normalized()
    st, ct = ti.sin(theta), ti.cos(theta)
    sp, cp = ti.sin(phi), ti.cos(phi)

    dr = st * cp * d.x + st * sp * d.y + ct * d.z
    dtheta = (ct * cp * d.x + ct * sp * d.y - st * d.z) / r
    dphi = (cp * d.y - sp * d.x) / (r * st)

    # Conserved Energy For A Null Geodesic (Photon)
    f = 1.0 - rs / r
    v_mag2 = dr**2 / f + r**2 * (dtheta**2 + st**2 * dphi**2)
    e_cons = ti.sqrt(v_mag2) * f

    return ti.Vector([r, theta, phi, dr, dtheta, dphi]), e_cons
