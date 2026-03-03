"""
Taichi-based Ray Tracer for Schwarzschild Black Hole
Renders gravitational lensing and accretion disk in real-time
"""

import math

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
        
        # Pixel buffer for rendered image
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        
        # Accumulation buffer for temporal anti-aliasing (optional)
        self.accum_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
        self.accum_count = ti.field(dtype=ti.i32, shape=())
        
    @ti.kernel
    def render(
        self,
        cam_pos: ti.template(),
        cam_fwd: ti.template(),
        cam_up: ti.template(),
        cam_right: ti.template(),
        fov: ti.f32,
        dl: ti.f32,
        max_steps: int,
    ):
        """
        Render the black hole scene from camera perspective
        """
        aspect = self.width / self.height
        tan_fov = ti.tan(fov / 2.0)
        
        for i, j in self.pixels:
            # Normalized device coordinates (-1 to 1)
            ndc_x = (2.0 * (i + 0.5) / self.width - 1.0)
            ndc_y = (2.0 * (j + 0.5) / self.height - 1.0)
            
            # Calculate ray direction
            px = ndc_x * aspect * tan_fov
            py = ndc_y * tan_fov
            
            direction = (cam_fwd + px * cam_right + py * cam_up).normalized()

            # Initialize ray state
            state, e_cons = init_ray(cam_pos, direction, self.rs)

            color = ti.Vector([0.0, 0.0, 0.0])
            curr_state = state

            # Disk hit flag and escape tracking
            disk_hit = False
            escaped = False
            r = curr_state[0]  # Track r for post-loop check

            for step in range(max_steps):
                r = curr_state[0]
                theta = curr_state[1]
                phi = curr_state[2]
                
                # Check if ray crossed event horizon
                if r <= self.rs * 1.01:
                    color = ti.Vector([0.0, 0.0, 0.0])  # Pure black (shadow)
                    break
                
                # Calculate z-coordinate (height above disk plane)
                curr_z = r * ti.cos(theta)
                
                # Step the ray
                next_state = rk4_step(curr_state, self.rs, e_cons, dl)
                next_z = next_state[0] * ti.cos(next_state[1])
                next_r = next_state[0]
                
                # Check for accretion disk intersection
                if not disk_hit:
                    # Detect disk plane crossing (z changes sign)
                    if (curr_z < 0 and next_z > 0) or (curr_z > 0 and next_z < 0):
                        # Check if within disk bounds (2.5rs to 6.5rs)
                        if 2.5 * self.rs < next_r < 6.5 * self.rs:
                            # Calculate disk color based on radius
                            # Inner disk is hotter (whiter), outer is cooler (redder)
                            t_param = (next_r / self.rs - 2.5) / 4.0  # 0 to 1
                            
                            # Temperature gradient: white-yellow-orange-red
                            r_col = 1.0
                            g_col = 0.7 - 0.3 * t_param
                            b_col = 0.2 * (1.0 - t_param)
                            
                            # Add brightness falloff
                            brightness = 1.5 * (1.0 - t_param * 0.5)
                            
                            color = ti.Vector([r_col, g_col, b_col]) * brightness
                            disk_hit = True
                            break
                
                # Check if ray escaped to infinity
                if next_r > 1e10:
                    escaped = True
                    # Generate starfield background based on final direction
                    star_phi = next_state[2]
                    star_theta = next_state[1]
                    
                    # Procedural starfield using trig functions for pseudo-randomness
                    noise = ti.sin(star_phi * 127.0) * ti.sin(star_theta * 89.0)
                    noise2 = ti.cos(star_phi * 73.0) * ti.cos(star_theta * 51.0)
                    
                    star_intensity = (noise + noise2) * 0.5 + 0.5
                    
                    # Star threshold - show stars
                    if star_intensity > 0.95:
                        color = ti.Vector([1.0, 1.0, 1.0]) * ((star_intensity - 0.95) * 40.0)
                    elif star_intensity > 0.85:
                        # Dim stars
                        hue = ti.sin(star_phi * 31.0) * 0.5 + 0.5
                        color = ti.Vector([0.7 + 0.3*hue, 0.7 + 0.3*(1-hue), 0.8]) * 0.3
                    else:
                        # Deep space background with slight nebula effect
                        nebula = ti.sin(star_phi * 17.0) * ti.cos(star_theta * 23.0)
                        color = ti.Vector([0.01 + 0.01*nebula, 0.015 + 0.01*nebula, 0.03 + 0.02*nebula])
                    
                    break
                
                curr_state = next_state
            
            # If we ran out of steps without escaping or hitting, it's likely captured
            if not escaped and not disk_hit and r > self.rs * 1.01:
                color = ti.Vector([0.0, 0.0, 0.0])
            
            self.pixels[i, j] = color
    
    @ti.kernel
    def render_aa(
        self,
        cam_pos: ti.template(),
        cam_fwd: ti.template(),
        cam_up: ti.template(),
        cam_right: ti.template(),
        fov: ti.f32,
        dl: ti.f32,
        max_steps: int,
        jitter_x: ti.f32,
        jitter_y: ti.f32,
    ):
        """
        Render with jittered sampling for anti-aliasing
        """
        aspect = self.width / self.height
        tan_fov = ti.tan(fov / 2.0)

        for i, j in self.pixels:
            # Add jitter for anti-aliasing
            ndc_x = (2.0 * (i + jitter_x) / self.width - 1.0)
            ndc_y = (2.0 * (j + jitter_y) / self.height - 1.0)
            
            px = ndc_x * aspect * tan_fov
            py = ndc_y * tan_fov
            
            direction = (cam_fwd + px * cam_right + py * cam_up).normalized()
            
            state, e_cons = init_ray(cam_pos, direction, self.rs)
            
            color = ti.Vector([0.0, 0.0, 0.0])
            curr_state = state
            
            for step in range(max_steps):
                r = curr_state[0]
                theta = curr_state[1]
                
                if r <= self.rs * 1.01:
                    break
                
                curr_z = r * ti.cos(theta)
                next_state = rk4_step(curr_state, self.rs, e_cons, dl)
                next_z = next_state[0] * ti.cos(next_state[1])
                next_r = next_state[0]
                
                # Accretion disk
                if (curr_z < 0 and next_z > 0) or (curr_z > 0 and next_z < 0):
                    if 2.5 * self.rs < next_r < 6.5 * self.rs:
                        t_param = (next_r / self.rs - 2.5) / 4.0
                        color = ti.Vector([1.0, 0.7 - 0.3 * t_param, 0.2 * (1.0 - t_param)]) * 1.5
                        break
                
                if next_r > 1e11:
                    star_phi = next_state[2]
                    star_theta = next_state[1]
                    noise = ti.sin(star_phi * 127.0) * ti.sin(star_theta * 89.0)
                    if noise > 0.97:
                        color = ti.Vector([1.0, 1.0, 1.0])
                    else:
                        color = ti.Vector([0.005, 0.008, 0.015])
                    break
                
                curr_state = next_state
            
            # Accumulate
            idx = i, j
            self.accum_buffer[idx] += color
            self.accum_count[None] += 1
    
    @ti.kernel
    def get_accumulated_image(self) -> ti.types.ndarray():
        """Get the accumulated image divided by frame count"""
        count = self.accum_count[None]
        if count == 0:
            count = 1
        for i, j in self.pixels:
            self.pixels[i, j] = self.accum_buffer[i, j] / count
    
    def reset_accumulation(self):
        """Reset accumulation buffer for new camera position"""
        self.accum_buffer.fill(0)
        self.accum_count[None] = 0
    
    def get_image(self):
        """Get rendered image as numpy array"""
        return self.pixels.to_numpy()
    
    def get_accumulated_image_np(self):
        """Get accumulated image as numpy array"""
        self.get_accumulated_image()
        return self.pixels.to_numpy()
