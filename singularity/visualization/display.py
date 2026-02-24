import sys

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_display_manager(width, height):
    if is_colab():
        from .colab_display import ColabDisplay
        return ColabDisplay(width, height)
    else:
        from .pygame_window import PygameWindow
        return PygameWindow(width, height)
