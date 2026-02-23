import os

import taichi as ti

FORCE_CPU = os.environ.get("ARCH", "").lower() == "cpu"


def init_taichi():
    if not FORCE_CPU:
        try:
            ti.init(arch=ti.cuda)
            print("Taichi initialized with CUDA")
        except:
            ti.init(arch=ti.cpu)
            print("Taichi initialized with LLVM")
    else:
        ti.init(arch=ti.cpu)
        print("Taichi initialized with LLVM")


# Physical Constants
C = 299792458.0
G = 6.67430e-11
M = 8.54e36
rs = 2.0 * G * M / (C**2)

# Simulation Parameters
D_LAMBDA = 1e7
MAX_STEPS = 10000
ESCAPE_R = 1e14

# Render Settings
WIDTH = 200
HEIGHT = 150
