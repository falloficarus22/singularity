import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import PIL.Image
import io

class ColabDisplay:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.running = True
        self.handle = None

    def update(self, img_array):
        # img_array: (width, height, 3) from Taichi
        img = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        
        # Taichi (X, Y) -> Standard (Y, X) - transpose and flip
        img = np.transpose(img, (1, 0, 2))[::-1, :, :]
        
        f = io.BytesIO()
        PIL.Image.fromarray(img).save(f, 'png')
        
        if self.handle is None:
            self.handle = display(display_id=True)
        
        self.handle.update(PIL.Image.open(f))

    def handle_events(self):
        # On Colab, events (like keyboard) are harder to capture synchronously.
        # We'll use ipywidgets for this in a more advanced implementation.
        # For now, return empty dict-like structure to avoid crashing the main loop.
        class DummyKeys:
            def __getitem__(self, key): return False
        return DummyKeys()

    @property
    def clock(self):
        class DummyClock:
            def tick(self, fps): pass
        return DummyClock()
