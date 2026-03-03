"""
Singularity - Interactive Schwarzschild Black Hole Visualizer

Real-time gravitational lensing simulation using general relativity.
Supports both local (Pygame) and Google Colab (ipywidgets) environments.

Controls:
    WASD / Arrow Keys : Orbit camera around black hole
    Q / E             : Zoom in / out
    ESC               : Quit (local mode only)
"""

import math
import sys
import time

import numpy as np
import taichi as ti

try:
    import pygame
except ImportError:
    pygame = None

from singularity.config import (
    init_taichi,
    rs,
    WIDTH,
    HEIGHT,
    D_LAMBDA,
    MAX_STEPS,
    is_colab,
    IN_COLAB,
)
from singularity.renderer.raytracer import RayTracer
from singularity.renderer.camera import Camera
from singularity.visualization.display import get_display_manager


def print_controls():
    """Print control instructions to console"""
    print("\n" + "=" * 50)
    print("  SINGULARITY - Black Hole Visualizer")
    print("=" * 50)
    print("\nControls:")
    print("  W / ↑        : Orbit up")
    print("  S / ↓        : Orbit down")
    print("  A / ←        : Orbit left")
    print("  D / →        : Orbit right")
    print("  Q            : Zoom in")
    print("  E            : Zoom out")
    print("  ESC          : Quit (local mode)")
    print("\nPhysics:")
    print(f"  Black Hole   : Sagittarius A*")
    print(f"  Mass         : {8.54e36:.2e} kg (~4.3 million solar masses)")
    print(f"  r_s          : {rs:.3e} m ({rs / 1e9:.2f} billion km)")
    print(f"  Resolution   : {WIDTH} x {HEIGHT}")
    print("=" * 50 + "\n")


def run_colab_mode():
    """Run in Google Colab with ipywidgets controls"""
    print("Running in Colab mode...")
    
    # Initialize Taichi
    init_taichi()
    
    # Create renderer
    tracer = RayTracer(HEIGHT, WIDTH, rs)
    
    # Initialize camera
    cam_dist = 6.0 * rs
    camera = Camera(pos=[cam_dist, 0, 0.5 * rs], target=[0, 0, 0])
    
    # Get display manager (Colab)
    window = get_display_manager(WIDTH, HEIGHT)
    
    # Frame timing
    frame_count = 0
    start_time = time.time()
    
    print("Rendering started. Use the control buttons to navigate.")
    print("Close the notebook cell to stop.\n")
    
    # Main render loop
    while window.running:
        # Handle input from ipywidgets
        keys = window.handle_events()
        
        # Camera controls
        move_speed = 0.05
        moved = False
        
        # Check for key states
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            camera.orbit(-move_speed, 0)
            moved = True
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            camera.orbit(move_speed, 0)
            moved = True
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            camera.orbit(0, -move_speed)
            moved = True
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            camera.orbit(0, move_speed)
            moved = True
        if keys[pygame.K_q]:
            camera.zoom(0.95)
            moved = True
        if keys[pygame.K_e]:
            camera.zoom(1.05)
            moved = True
            
        if moved:
            tracer.reset_accumulation()
        
        # Get camera vectors for Taichi
        c_pos, c_fwd, c_up, c_rt = camera.get_taichi_vectors()
        
        # Render frame
        tracer.render(
            cam_pos=c_pos,
            cam_fwd=c_fwd,
            cam_up=c_up,
            cam_right=c_rt,
            fov=camera.fov,
            dl=D_LAMBDA,
            max_steps=MAX_STEPS
        )
        
        # Update display
        window.update(tracer.get_image())
        
        # Frame rate tracking
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f} | Distance: {camera.dist / rs:.2f} r_s")
            start_time = time.time()
            frame_count = 0
        
        window.clock.tick(30)  # Target 30 FPS for Colab
    
    print("Visualization stopped.")


def run_local_mode():
    """Run locally with Pygame window"""
    print("Running in local mode...")
    
    try:
        import pygame
    except ImportError:
        print("Error: pygame not installed. Install with: pip install pygame")
        sys.exit(1)
    
    # Initialize Taichi
    init_taichi()
    
    # Create renderer
    tracer = RayTracer(HEIGHT, WIDTH, rs)
    
    # Initialize camera
    cam_dist = 6.0 * rs
    camera = Camera(pos=[cam_dist, 0, 0.5 * rs], target=[0, 0, 0])
    
    # Get display manager (Pygame)
    window = get_display_manager(WIDTH, HEIGHT)
    
    # Frame timing
    frame_count = 0
    start_time = time.time()
    
    # Main render loop
    while window.running:
        # Handle Pygame events
        keys = window.handle_events()
        
        # Camera controls
        move_speed = 0.05
        moved = False
        
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            camera.orbit(-move_speed, 0)
            moved = True
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            camera.orbit(move_speed, 0)
            moved = True
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            camera.orbit(0, -move_speed)
            moved = True
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            camera.orbit(0, move_speed)
            moved = True
        if keys[pygame.K_q]:
            camera.zoom(0.95)
            moved = True
        if keys[pygame.K_e]:
            camera.zoom(1.05)
            moved = True
            
        if moved:
            tracer.reset_accumulation()
        
        # Get camera vectors for Taichi
        c_pos, c_fwd, c_up, c_rt = camera.get_taichi_vectors()
        
        # Render frame
        tracer.render(
            cam_pos=c_pos,
            cam_fwd=c_fwd,
            cam_up=c_up,
            cam_right=c_rt,
            fov=camera.fov,
            dl=D_LAMBDA,
            max_steps=MAX_STEPS
        )
        
        # Update display
        window.update(tracer.get_image())
        
        # Frame rate tracking
        frame_count += 1
        if frame_count % 60 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f} | Distance: {camera.dist / rs:.2f} r_s")
            start_time = time.time()
            frame_count = 0
        
        window.clock.tick(60)  # Target 60 FPS
    
    pygame.quit()
    print("Visualization stopped.")


def main():
    """Main entry point"""
    try:
        print_controls()
        
        # Detect environment and run appropriate mode
        if is_colab() or IN_COLAB:
            run_colab_mode()
        else:
            run_local_mode()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
