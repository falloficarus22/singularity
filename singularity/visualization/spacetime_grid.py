import taichi as ti
import numpy as np

@ti.data_oriented
class SpacetimeGrid:
    def __init__(self, rs, size=30, resolution=50):
        self.rs = rs
        self.size = size
        self.res = resolution
        
    @ti.func
    def get_flamm_height(self, r):
        # Flamm's Paraboloid: z = 2 * sqrt(rs * (r - rs))
        h = 0.0
        if r > self.rs:
            h = 2.0 * ti.sqrt(self.rs * (r - self.rs))
        return h

    @ti.func
    def intersect_grid(self, pos, d, dl):
        # This is a complex task for a simple raytracer.
        # Often we just project the grid onto the disk plane 
        # or use it as a separate visualization mode.
        # For Phase 3, we'll implement a simple wireframe overlay logic.
        pass
