from settings import *
import pygame

class Circle(pygame.sprite.Sprite):
    def __init__(self, game, x, y, r):
        self.groups = game.shape_sprites
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.x = x
        self.y = y
        self.r = r
        self.image = pygame.Surface((WIDTH, HEIGHT))
        self.image.fill(BACKGROUND_COLOR)
        pygame.draw.circle(self.image, YELLOW, (self.x, self.y), self.r, 2)
        self.rect = self.image.get_rect()