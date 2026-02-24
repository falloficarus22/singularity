import pygame
import numpy as np
import taichi as ti

class PygameWindow:
    def __init__(self, width, height, title="Singularity"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.running = True

    def update(self, img_array):
        # img_array is (width, height, 3) from Taichi
        # Pygame expects (width, height, 3) but often needs it to be in its surface format
        # and usually vertical flip
        img = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        
        # Taichi (X, Y) -> Pygame expects (X, Y)
        # However, we usually want (0,0) at top-left. Taichi's (0,0) is bottom-left usually in how we index.
        # Let's flip the Y axis for display.
        img = np.flip(img, axis=1)
        
        surface = pygame.surfarray.make_surface(img)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
        return pygame.key.get_pressed()
