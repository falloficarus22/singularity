import numpy as np
import taichi as ti

from singularity.physics.integrator import rk4_step
from singularity.physics.ray import init_ray


@ti.data_oriented
class RayTracer:
    def __init__(self, height, width, rs):
        self.height = height
        self.width = width
        self.rs = rs
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

    @ti.kernel
    def render(
        self,
        cam_pos: ti.types.vector(3, ti.f32),
        cam_fwd: ti.types.vector(3, ti.f32),
        cam_up: ti.types.vector(3, ti.f32),
        cam_right: ti.types.vector(3, ti.f32),
        fov: ti.f32,
        dl: ti.f32,
        max_steps: int,
    ):
        for i, j in self.pixels:
            aspect = self.width / self.height
            px = (2.0 * (i + 0.5) / self.width - 1.0) * aspect * ti.tan(fov / 2.0)
            py = (2.0 * (j + 0.5) / self.height - 1.0) * ti.tan(fov / 2.0)

            direction = (cam_fwd + px * cam_right + py * cam_up).normalized()

            # Initial Ray State
            state, e_cons = init_ray(cam_pos, direction, self.rs)

            color = ti.Vector([0.0, 0.0, 0.0])
            curr_state = state

            for step in range(max_steps):
                # Capture Check
                if curr_state[0] <= self.rs * 1.01:
                    break

                # Accretion Disk
                prev_z = curr_state[0] * ti.cos(curr_state[1])
                next_state = rk4_step(curr_state, self.rs, e_cons, dl)
                next_z = next_state[0] * ti.cos(next_state[1])

                if (prev_z < 0 and next_z > 0) or (prev_z > 0 and next_z < 0):
                    r_hit = next_state[0]
                    if 2.5 * self.rs < r_hit < 6.5 * self.rs:
                        # Simple hot fire-colored disl
                        temp = 1.0 / (r_hit / self.rs)
                        color = ti.Vector([1.0, 0.5 * temp, 0.1 * temp]) * 1.5
                        break

                # Escape Check
                if next_state[0] > 1e11:
                    # Basic Starfield Background
                    theta, phi = next_state[1], next_state[2]
                    star = ti.sin(phi * 60) * ti.sin(theta * 60)

                    if star > 0.99:
                        color = ti.Vector([1.0, 1.0, 1.0])
                    else:
                        color = ti.Vector([0.01, 0.01, 0.04])

                    break

                curr_state = next_state

            self.pixels[i, j] = color

    def get_image(self):
        return self.pixels.to_numpy()
