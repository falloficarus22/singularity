# Singularity - Black Hole Visualizer

Python-based **interactive black hole simulation** that visualizes gravitational lensing around a Schwarzschild black hole using general relativity.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Taichi](https://img.shields.io/badge/Taichi-1.7+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Real-time gravitational lensing** - See light bend around a black hole
- **Schwarzschild metric** - Accurate general relativity simulation for non-rotating black holes
- **Interactive camera** - Orbit and zoom around the black hole
- **Accretion disk visualization** - Hot inner disk glowing white-yellow, cooler outer regions orange-red
- **Procedural starfield** - Dynamic background stars
- **Dual platform support** - Works on local machines (Pygame) and Google Colab (ipywidgets)
- **GPU accelerated** - Uses Taichi for CUDA/Vulkan/OpenGL compute when available

## Physics Model

Simulates **Sagittarius A*** (Sgr A*), the supermassive black hole at the center of the Milky Way:

| Parameter | Value |
|-----------|-------|
| Mass | 8.54×10³⁶ kg (~4.3 million solar masses) |
| Schwarzschild radius (r_s) | ~1.27×10¹⁰ m |
| Photon sphere | 1.5 r_s |
| ISCO | 3 r_s |
| Accretion disk | 2.5 - 6.5 r_s |

## Installation

### Requirements

- Python 3.10+
- Taichi 1.7+
- NumPy
- Pygame (for local mode)

### Install Dependencies

```bash
# Core dependencies
pip install taichi numpy

# For local display
pip install pygame

# For Colab (optional - pre-installed)
pip install ipywidgets matplotlib Pillow
```

## Usage

### Local Mode (Pygame Window)

```bash
python app.py
```

### Google Colab Mode

The app automatically detects Colab environment. Simply run:

```python
!pip install taichi numpy pygame ipywidgets
%run app.py
```

### Force CPU Mode

```bash
ARCH=cpu python app.py
```

## Controls

| Key | Action |
|-----|--------|
| **W / ↑** | Orbit camera up |
| **S / ↓** | Orbit camera down |
| **A / ←** | Orbit camera left |
| **D / →** | Orbit camera right |
| **Q** | Zoom in |
| **E** | Zoom out |
| **ESC** | Quit (local mode only) |

### Colab Controls

In Google Colab, interactive buttons appear below the visualization for camera control.

## Project Structure

```
singularity/
├── app.py                      # Main application entry point
├── geodesic.py                 # Standalone geodesic tracer (matplotlib plots)
├── singularity/
│   ├── config.py               # Physical constants, simulation parameters
│   ├── physics/
│   │   ├── schwarzschild.py    # Schwarzschild metric equations
│   │   ├── integrator.py       # RK4 integration kernel
│   │   └── ray.py              # Ray initialization
│   ├── renderer/
│   │   ├── raytracer.py        # Taichi ray tracing kernel
│   │   └── camera.py           # Camera orbit/zoom controls
│   └── visualization/
│       ├── display.py          # Display manager factory
│       ├── pygame_window.py    # Local Pygame window
│       └── colab_display.py    # Colab ipywidgets interface
```

## How It Works

1. **Ray Initialization** - For each pixel, cast a ray from the camera
2. **Geodesic Integration** - Trace the ray through curved spacetime using 4th-order Runge-Kutta
3. **Disk Intersection** - Detect when rays cross the accretion disk plane
4. **Color Calculation** - Assign colors based on disk temperature or starfield position
5. **Real-time Rendering** - Update at 30-60 FPS depending on hardware

## Performance

| Platform | Resolution | FPS |
|----------|------------|-----|
| CUDA GPU | 640×480 | 30-60 |
| Vulkan GPU | 640×480 | 20-40 |
| CPU (LLVM) | 320×240 | 1-5 |

Performance varies based on step count and hardware. GPU acceleration is strongly recommended.

## Additional Scripts

### Geodesic Plotter

Generate static 2D/3D plots of photon geodesics:

```bash
python geodesic.py
```

Outputs:
- `geodesics_2d.png` - 2D projection of ray paths
- `geodesics_3d.png` - 3D visualization

## Troubleshooting

### "No module named 'pygame'"
```bash
pip install pygame
```

### "Taichi initialized with CPU fallback"
GPU backend not available. Install CUDA drivers or use Vulkan:
```bash
# For NVIDIA GPUs
pip install --upgrade taichi

# Force specific backend
python -c "import taichi as ti; ti.init(arch=ti.vulkan)"
```

### Slow rendering in Colab
Reduce resolution in `config.py`:
```python
WIDTH, HEIGHT = 320, 240  # Lower resolution
```

## Credits

Inspired by [Kavan's black_hole](https://github.com/kavan010/black_hole)

## License

MIT License - See LICENSE file for details
