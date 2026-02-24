import numpy as np
import taichi as ti

@ti.data_oriented
class Camera:
    def __init__(self, pos, target, up=np.array([0, 0, 1]), fov=60):
        self.pos = np.array(pos, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up_dir = np.array(up, dtype=np.float32)
        self.fov = np.radians(fov)
        
        self.dist = np.linalg.norm(self.pos - self.target)
        self.theta = np.arctan2(self.pos[1], self.pos[0])
        self.phi = np.arccos(self.pos[2] / self.dist)
        
        self.update_vectors()

    def update_vectors(self):
        # Calculate Forward, Right, Up vectors for the raytracer
        fwd = (self.target - self.pos)
        self.fwd = fwd / np.linalg.norm(fwd)
        
        right = np.cross(self.fwd, self.up_dir)
        self.right = right / np.linalg.norm(right)
        
        up = np.cross(self.right, self.fwd)
        self.up = up / np.linalg.norm(up)

    def orbit(self, d_theta, d_phi):
        self.theta += d_theta
        self.phi = np.clip(self.phi + d_phi, 0.01, np.pi - 0.01)
        
        self.pos[0] = self.target[0] + self.dist * np.sin(self.phi) * np.cos(self.theta)
        self.pos[1] = self.target[1] + self.dist * np.sin(self.phi) * np.sin(self.theta)
        self.pos[2] = self.target[2] + self.dist * np.cos(self.phi)
        
        self.update_vectors()

    def zoom(self, factor):
        self.dist *= factor
        # Minimum distance to avoid crossing rs too easily
        self.dist = max(self.dist, 3.0 * 1.269e10) # 3 rs
        
        self.pos[0] = self.target[0] + self.dist * np.sin(self.phi) * np.cos(self.theta)
        self.pos[1] = self.target[1] + self.dist * np.sin(self.phi) * np.sin(self.theta)
        self.pos[2] = self.target[2] + self.dist * np.cos(self.phi)
        
        self.update_vectors()

    def get_taichi_vectors(self):
        return (
            ti.Vector(self.pos),
            ti.Vector(self.fwd),
            ti.Vector(self.up),
            ti.Vector(self.right)
        )
