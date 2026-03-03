import os

import taichi as ti

# Environment detection
FORCE_CPU = os.environ.get("ARCH", "").lower() == "cpu"
IN_COLAB = "COLAB_GPU" in os.environ


def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab

        return True
    except ImportError:
        return "COLAB_GPU" in os.environ


def get_render_resolution():
    """Get appropriate resolution based on platform"""
    if IN_COLAB or is_colab():
        # Colab: moderate resolution for faster rendering
        return 400, 300
    elif FORCE_CPU:
        # CPU-only: moderate resolution for performance
        return 100, 75
    else:
        # GPU: higher resolution
        return 640, 480


def init_taichi(arch=None):
    """
    Initialize Taichi with appropriate backend

    Args:
        arch: Optional architecture override ('cuda', 'cpu', 'vulkan', 'opengl')
    """
    if arch:
        if arch == "cuda":
            ti.init(arch=ti.cuda)
            print("Taichi initialized with CUDA")
        elif arch == "vulkan":
            ti.init(arch=ti.vulkan)
            print("Taichi initialized with Vulkan")
        elif arch == "opengl":
            ti.init(arch=ti.opengl)
            print("Taichi initialized with OpenGL")
        else:
            ti.init(arch=ti.cpu)
            print("Taichi initialized with LLVM (CPU)")
        return

    if not FORCE_CPU:
        # Try GPU backends in order of preference
        gpu_backends = [
            ("cuda", ti.cuda, "CUDA"),
            ("vulkan", ti.vulkan, "Vulkan"),
            ("opengl", ti.opengl, "OpenGL"),
        ]

        for name, arch_const, display_name in gpu_backends:
            try:
                ti.init(arch=arch_const)
                print(f"Taichi initialized with {display_name}")
                return
            except Exception:
                continue

        # Fall back to CPU
        ti.init(arch=ti.cpu)
        print("Taichi initialized with LLVM (CPU fallback)")
    else:
        ti.init(arch=ti.cpu)
        print("Taichi initialized with LLVM (CPU forced)")


# Physical Constants - Sagittarius A* Black Hole
C = 299792458.0  # Speed of light (m/s)
G = 6.67430e-11  # Gravitational constant (m³/kg·s²)
M = 8.54e36  # Mass of Sgr A* (kg) ~ 4.3 million solar masses
rs = 2.0 * G * M / (C**2)  # Schwarzschild radius

# Simulation Parameters
D_LAMBDA = 1e7  # Integration step size
MAX_STEPS = 10000  # Maximum ray tracing steps
ESCAPE_R = 1e14  # Escape distance (m)

# Render Settings (dynamically set based on platform)
WIDTH, HEIGHT = get_render_resolution()
