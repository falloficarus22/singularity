import sys
import numpy as np
import taichi as ti
import pygame
from singularity.config import init_taichi, rs, WIDTH, HEIGHT, D_LAMBDA, MAX_STEPS
from singularity.renderer.raytracer import RayTracer
from singularity.renderer.camera import Camera
from singularity.visualization.display import get_display_manager

def main():
    # Initialize Taichi
    init_taichi()
    
    # Initialize Renderer
    tracer = RayTracer(HEIGHT, WIDTH, rs)
    
    # Initialize Camera
    cam_dist = 6.0 * rs
    camera = Camera(pos=[cam_dist, 0, 0.5 * rs], target=[0, 0, 0])
    
    # Initialize Window
    window = get_display_manager(WIDTH, HEIGHT)
    
    print("Controls:")
    print("  WASD / Arrow Keys : Orbit Camera")
    print("  Q / E             : Zoom In / Out")
    print("  ESC               : Quit")
    
    while window.running:
        keys = window.handle_events()
        
        # Camera Controls
        move_speed = 0.05
        
        # Check for Pygame keys if we are in local mode
        try:
            import pygame
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                camera.orbit(-move_speed, 0)
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                camera.orbit(move_speed, 0)
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                camera.orbit(0, -move_speed)
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                camera.orbit(0, move_speed)
            if keys[pygame.K_q]:
                camera.zoom(0.95)
            if keys[pygame.K_e]:
                camera.zoom(1.05)
        except (ImportError, AttributeError, KeyError):
            pass
            
        # Get Camera Vectors for Taichi
        c_pos, c_fwd, c_up, c_rt = camera.get_taichi_vectors()
        
        # Render Frame
        tracer.render(
            cam_pos=c_pos,
            cam_fwd=c_fwd,
            cam_up=c_up,
            cam_right=c_rt,
            fov=camera.fov,
            dl=D_LAMBDA,
            max_steps=MAX_STEPS
        )
        
        # Update Display
        window.update(tracer.get_image())
        window.clock.tick(60)

    if pygame:
        pygame.quit()

if __name__ == "__main__":
    main()
