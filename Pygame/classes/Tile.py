import Pygame


class Tile:
    def __init__(self, sprite, rect):
        self.sprite = sprite
        self.rect = rect

    def drawRect(self, screen):
        try:
            Pygame.draw.rect(screen, Pygame.Color(255, 0, 0), self.rect, 1)
        except Exception:
            pass
