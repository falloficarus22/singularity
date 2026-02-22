"""
Schwarzschild Geodesic Tracer
Traces photon path through curved spacetime using 4th order Runge-Kutta method.
Theory: Schwarzschild metric (non-rotating black hole)
Black Hole: Sagittarius A*
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.typing import RGBAColorType, RGBAColourType, RGBColorType
from mpl_toolkits.mplot3d import Axes3D
from pandas.io.formats.printing import EscapeChars

# Physical Constants & Black Hole Parameters
C = 299792458.0
G = 6.67430e-11
M = 8.54e36
r_s = 2.0 * G * M / (C**2)

# Integration Parameters
D_LAMBDA = 1e7
MAX_STEPS = 60000
ESCAPE_R = 1e14


# Ray Initialization
def init_ray(pos, direction):
    x, y, z = pos

    # Spherical Coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    # Direction
    d = direction / np.linalg.norm(direction)
    dx, dy, dz = d

    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    dr = st * cp * dx + st * sp * dy + ct * dz
    dtheta = ct * cp * dx / r + ct * sp * dy / r + st * dz / r
    dphi = -(sp * dx / (r * st)) + cp * dy / (r * st)

    # Conserved quantities
    L = r**2 * st * dphi
    f = 1.0 - r_s / r
    dt_dl = np.sqrt(dr**2 / f + r**2 * (dtheta**2 + st**2 * dphi**2))
    E = f * dt_dl

    state = np.array([r, theta, phi, dr, dtheta, dphi])
    return state, E, L


# Geodesic Equations
def geodesic_rhs(state, E):
    r, theta, phi, dr, dtheta, dphi = state

    st = np.sin(theta)
    ct = np.cos(theta)

    f = 1.0 - r_s / r
    dt_dl = E / f

    # Accelerations
    ddr = (
        -(r_s / (2.0 * r**2)) * f * dt_dl**2
        + (r_s / (2.0 * r**2 * f)) * dr**2
        + r * (dtheta**2 + st**2 * dphi**2)
    )

    ddtheta = -2.0 * dr * dtheta / r + st * ct * dphi**2

    ddphi = -2.0 * dr * dphi / r - 2.0 * (ct / st) * dtheta * dphi

    return np.array([dr, dtheta, dphi, ddr, ddtheta, ddphi])


# RK4 Integrator
def rk4_step(state, E, dl):
    k1 = geodesic_rhs(state, E)
    k2 = geodesic_rhs(state + dl / 2.0 * k1, E)
    k3 = geodesic_rhs(state + dl / 2.0 * k2, E)
    k4 = geodesic_rhs(state + dl * k3, E)

    return state + (dl / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# Spherical -> Cartesian Conversion
def to_cartesian(state):
    r, theta, phi = state[0], state[1], state[2]
    return np.array(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    )


# Ray Tracing
def trace_ray(pos, direction, dl=D_LAMBDA, max_steps=MAX_STEPS):
    state, E, L = init_ray(pos, direction)
    trail = [to_cartesian(state)]
    fate = "max_steps"

    r_start = state[0]

    for step in range(max_steps):
        if state[0] <= r_s:
            fate = "captured"
            break

        state = rk4_step(state, E, dl)

        trail.append(to_cartesian(state))

        r_current = state[0]
        dr_current = state[3]
        if r_current > r_start and dr_current > 0:
            fate = "escaped"
            break

    return np.array(trail), fate


# Visualization (2D)
def plot_geodesics_2d(rays_data):
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    # Event Horizon
    eh = Circle((0, 0), r_s, color="black", ec="#ff4444", linewidth=2, zorder=10)
    ax.add_patch(eh)

    # Photon Sphere
    ps = Circle(
        (0, 0),
        1.5 * r_s,
        fill=False,
        ec="#44ddff",
        linewidth=1,
        linestyle="--",
        alpha=0.4,
        zorder=5,
    )
    ax.add_patch(ps)

    # Innermost Stable Circular Orbit
    isco = Circle(
        (0, 0),
        3.0 * r_s,
        fill=False,
        ec="#44aaff",
        linewidth=1,
        linestyle=":",
        alpha=0.3,
        zorder=5,
    )
    ax.add_patch(isco)

    # Accretion Disk Edges
    for r_disk in [r_s * 2.2, r_s * 5.2]:
        disk = Circle(
            (0, 0),
            r_disk,
            fill=False,
            ec="#ff8800",
            linewidth=0.8,
            linestyle=":",
            alpha=0.25,
            zorder=5,
        )
    ax.add_patch(disk)

    # Ray Trajectories
    n = len(rays_data)
    for i, (trail, fate) in enumerate(rays_data):
        # Color based on impact parameter
        t = i / max(n - 1, 1)
        if fate == "captured":
            color = "#ff3333"
            alpha = 0.6
        elif fate == "escaped":
            color = plt.cm.cool(t)
            alpha = 0.85
        else:
            color = "#888888"
            alpha = 0.4

        # Subsample long trails for plotting speed
        step = max(1, len(trail) // 3000)
        x, z = trail[::step, 0], trail[::step, 2]
        ax.plot(x, z, color=color, linewidth=0.7, alpha=alpha)

    scale = 6 * r_s
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color="white", fontsize=12)
    ax.set_ylabel("z (m)", color="white", fontsize=12)
    ax.set_title(
        "Schwartzschild Geodesics = Light Bending Around a Black Hole",
        color="white",
        fontsize=12,
        pad=15,
    )
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#ff4444",
            linewidth=2,
            label=f"Event Horizon  r_s = {r_s:.3e} m",
        ),
        Line2D(
            [0],
            [0],
            color="#ffdd44",
            linewidth=1,
            linestyle="--",
            label="Photon Sphere  r = 1.5 r_s",
        ),
        Line2D(
            [0],
            [0],
            color="#44aaff",
            linewidth=1,
            linestyle=":",
            label="ISCO  r = 3 r_s",
        ),
        Line2D([0], [0], color="#ff3333", linewidth=1.5, label="Captured rays"),
        Line2D([0], [0], color="#44ddff", linewidth=1.5, label="Escaped rays"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=9,
        facecolor="#1a1f2e",
        edgecolor="#444444",
        labelcolor="white",
    )

    plt.tight_layout()
    plt.savefig(
        "geodesics_2d.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight"
    )
    plt.show()
    print("Image saved")


# Visualization (3D)
def plot_geodesics_3d(rays_data):
    fig = plt.figure(figsize=(12, 10), facecolor="#0d1117")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0d1117")

    # Event Horizon Sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = r_s * np.outer(np.cos(u), np.sin(v))
    ys = r_s * np.outer(np.sin(u), np.sin(v))
    zs = r_s * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="black", alpha=0.95, zorder=0)

    # Ray Trajectories
    n = len(rays_data)
    for i, (trail, fate) in enumerate(rays_data):
        t = i / max(n - 1, 1)
        color = "#ff3333" if fate == "captured" else plt.cm.cool(t)
        alpha = 0.5 if fate == "captured" else 0.75

        step = max(1, len(trail) // 1500)
        ax.plot(
            trail[::step, 0],
            trail[::step, 1],
            trail[::step, 2],
            color=color,
            linewidth=0.5,
            alpha=alpha,
        )

    limit = 5 * r_s
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.set_zlabel("z", color="white")
    ax.set_title("3D Schwarzschild Geodesics", color="white", fontsize=14, pad=15)
    ax.tick_params(colors="#666666")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(
        "geodesics_3d.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight"
    )
    plt.show()
    print("3D Image saved")


# Main
def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       SINGULARITY — Schwarzschild Geodesic Tracer        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f" Black Hole : Sagittarius A*")
    print(f" Mass       : {M:.2e} kg")
    print(f" r_s        : {r_s:.4e} m ({r_s / 1e9:.2f} x 10^9 m)")
    print(f" Step (dʎ)  : {D_LAMBDA:.0e}")
    print(f" Max Steps  : {MAX_STEPS}")
    print()

    # Camera Setup
    cam_distance = 6.34194e10
    cam_pos = np.array([cam_distance, 0.0, 0.0])

    print(f" Camera     : [{cam_distance:.2e}, 0, 0] m")
    print(f"              ({cam_distance / r_s:.1f} r_s from center)")
    print()

    # Fire Rays
    num_rays = 30
    offsets = np.linspace(-4.0 * r_s, 4.0 * r_s, num_rays)

    print(f" Tracing {num_rays} rays...")
    print(
        f" Impact Parameters: {offsets[0] / r_s:+.1f} r_s -> {offsets[-1] / r_s:+.1f} r_s"
    )
    print()

    rays_data = []
    t0 = time.time()

    for i, offset in enumerate(offsets):
        # Aim toward the origin with z-offset
        target = np.array([0.0, 0.0, offset])
        direction = target - cam_pos

        trail, fate = trace_ray(cam_pos, direction)
        rays_data.append((trail, fate))

        elapsed = time.time() - t0
        print(
            f" Ray {i + 1:3d}/{num_rays} | b = {offset / r_s:+6.2f} r_s | "
            f"{fate:10s} | {len(trail):6d} steps | {elapsed:.1f}s"
        )

    total_time = time.time() - t0

    # Summary
    captured = sum(1 for _, f in rays_data if f == "captured")
    escaped = sum(1 for _, f in rays_data if f == "escaped")
    print()
    print(f" ---------------------- Summary -----------------------")
    print(f" Captured : {captured} rays (fell into the black hole)")
    print(f" Escaped  : {escaped} rays (deflected and flew away)")
    print(f" Total    : {total_time:.1f} seconds")
    print()

    # Plot
    print(" Generating plots...")
    plot_geodesics_2d(rays_data)
    plot_geodesics_3d(rays_data)
    print()


if __name__ == "__main__":
    main()
