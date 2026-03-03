"""
Colab Display Manager with Interactive Controls
Uses ipywidgets for keyboard-like controls in Google Colab
"""

import numpy as np
import PIL.Image
import io
from IPython.display import display, clear_output
import ipywidgets as widgets
from IPython.display import display as ipy_display


class ColabDisplay:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.running = True
        self.handle = None

        # Camera control state
        self.keys = {
            "w": False,
            "s": False,
            "a": False,
            "d": False,
            "q": False,
            "e": False,
            "up": False,
            "down": False,
            "left": False,
            "right": False,
        }

        # Create control buttons
        self._create_controls()

    def _create_controls(self):
        """Create interactive control buttons for Colab"""
        # Button style
        button_style = {"width": "50px", "height": "50px"}

        # Direction buttons
        self.btn_up = widgets.Button(description="↑", **button_style)
        self.btn_down = widgets.Button(description="↓", **button_style)
        self.btn_left = widgets.Button(description="←", **button_style)
        self.btn_right = widgets.Button(description="→", **button_style)

        # Zoom buttons
        self.btn_zoom_in = widgets.Button(
            description="Zoom In", style={"button_color": "#4CAF50"}
        )
        self.btn_zoom_out = widgets.Button(
            description="Zoom Out", style={"button_color": "#f44336"}
        )

        # Quit button
        self.btn_quit = widgets.Button(
            description="QUIT", style={"button_color": "#ff0000"}
        )

        # Bind events
        self.btn_up.on_click(lambda b: self._set_key("up", True))

        self.btn_down.on_click(lambda b: self._set_key("down", True))

        self.btn_left.on_click(lambda b: self._set_key("left", True))

        self.btn_right.on_click(lambda b: self._set_key("right", True))

        self.btn_zoom_in.on_click(lambda b: self._set_key("q", True))
        self.btn_zoom_out.on_click(lambda b: self._set_key("e", True))

        self.btn_quit.on_click(lambda b: self._quit())

        # Layout
        nav_box = widgets.HBox(
            [
                widgets.VBox([self.btn_up, widgets.Label(value="Orbit Up")]),
                widgets.VBox([self.btn_left, widgets.Label(value="Left")]),
                widgets.VBox([self.btn_down, widgets.Label(value="Down")]),
                widgets.VBox([self.btn_right, widgets.Label(value="Right")]),
            ]
        )

        zoom_box = widgets.HBox([self.btn_zoom_in, self.btn_zoom_out])

        quit_box = widgets.HBox([self.btn_quit])

        controls = widgets.VBox(
            [
                widgets.HTML(value="<h4>Camera Controls</h4>"),
                nav_box,
                zoom_box,
                quit_box,
                widgets.HTML(
                    value="<p><b>Instructions:</b> Click arrows to orbit, zoom buttons to move closer/farther</p>"
                ),
            ]
        )

        ipy_display(controls)

    def _set_key(self, key, value):
        self.keys[key] = value

    def _release_key(self, key):
        self.keys[key] = False

    def _quit(self):
        self.running = False

    def update(self, img_array):
        """Update the displayed image"""
        # img_array: (width, height, 3) from Taichi
        img = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)

        # Taichi (X, Y) -> Standard (Y, X) - transpose and flip
        img = np.transpose(img, (1, 0, 2))[::-1, :, :]

        f = io.BytesIO()
        PIL.Image.fromarray(img).save(f, "png")

        if self.handle is None:
            self.handle = display(display_id=True)

        self.handle.update(PIL.Image.open(f))

    def handle_events(self):
        """Return current key state for camera control"""

        # Map button keys to pygame-like key codes
        class KeyState:
            def __init__(self, keys):
                self.keys = keys

            def __getitem__(self, key):
                # Map pygame constants to our keys
                import pygame

                key_map = {
                    pygame.K_w: "w",
                    pygame.K_s: "s",
                    pygame.K_a: "a",
                    pygame.K_d: "d",
                    pygame.K_q: "q",
                    pygame.K_e: "e",
                    pygame.K_UP: "up",
                    pygame.K_DOWN: "down",
                    pygame.K_LEFT: "left",
                    pygame.K_RIGHT: "right",
                }
                mapped = key_map.get(key, None)
                if mapped:
                    return self.keys.get(mapped, False)
                return False

        return KeyState(self.keys)

    @property
    def clock(self):
        class DummyClock:
            def tick(self, fps):
                pass

        return DummyClock()
