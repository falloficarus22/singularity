import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

from config import D_LAMBDA, HEIGHT, MAX_STEPS, WIDTH, init_taichi, rs
from renderer.raytracer import RayTracer


def main():
    init_taichi()

    tracer = RayTracer(HEIGHT, WIDTH, rs)

    # Camera: Orbiting Slightly Tilted
    cam_pos = ti.Vector([6.0 * rs, 0.0, 1.0 * rs])
    cam_fwd = (ti.Vector([0.0, 0.0, 0.0]) - cam_pos).normalized()
    cam_rt = ti.Vector([0.0, 1.0, 0.0])
    cam_up = cam_rt.cross(cam_fwd).normalized()

    fov = np.radians(60)

    print(f"Starting Renderer ({WIDTH}x{HEIGHT})...")
    tracer.render(cam_pos, cam_fwd, cam_up, cam_rt, fov, D_LAMBDA, MAX_STEPS)

    img = tracer.get_image()
    img = np.transpose(img, (1, 0, 2))[::-1, :, :]

    plt.figure(figsize=(10, 7.5), facecolor="black")
    plt.imshow(img)
    plt.axis("off")
    plt.title("SINGULARITY - Taichi Render", color="white")
    plt.savefig("phase2.png", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
