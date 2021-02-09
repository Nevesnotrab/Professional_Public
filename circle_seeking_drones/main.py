import pygame, sys
import random
from math import sqrt
from settings import *
from drone import Drone
from circle import Circle

class Game:
    def __init__(self):
        self.running = True
        self.display_surf = None
        self.FPS = 60
        self.FramePerSec = pygame.time.Clock()
        self.dt = 0.
        self.drone_sprites = None
        self.player = None
        self.drone_array = []
        self.circle_array = []
        self.circle_present = False
        self.circlex = None
        self.circley = None
        self.circler = None

    def on_init(self):
        pygame.init()
        self.drone_sprites = pygame.sprite.Group()
        self.shape_sprites = pygame.sprite.Group()
        self.display_surf = pygame.display.set_mode(win_size)
        pygame.display.set_caption(GAME_TITLE)
        for i in range(0, SWARM_SIZE):
            self.drone_array.append(Drone(self, random.randint(0, WIDTH - DRONE_SIZE), random.randint(0, HEIGHT - DRONE_SIZE)))
        pygame.display.update()
        self.load_data()

    def on_event(self, event):
        if  (event.type == pygame.QUIT):
            self.running = False
        if  (event.type == pygame.KEYDOWN):
            if  (event.key == pygame.K_ESCAPE):
                self.running = False
            if  (event.key == pygame.K_n):
                self.draw_circle()

    def pythagorean(self, x1, x2, y1, y2):
        #Ye Pythagorean theorem
        return sqrt(((x1 - x2) ** 2.) + ((y1 - y2) ** 2.))

    def calculate_drone_closeness(self):
        for i in range(0, SWARM_SIZE):
            #distance is declared larger than the maximum distance between two drones
            distance = int(self.pythagorean(0, WIDTH, 0, HEIGHT) + 1)
            pos = (0, 0)
            for j in range(0, SWARM_SIZE):
                if (i != j):
                    d1pos = (self.drone_array[i].x, self.drone_array[i].y)
                    d2pos = (self.drone_array[j].x, self.drone_array[j].y)
                    new_dist = self.pythagorean(d1pos[0], d2pos[0], d1pos[1], d2pos[1])
                    if (new_dist < distance):
                        distance = new_dist
                        pos = (self.drone_array[j].x, self.drone_array[j].y)
            self.drone_array[i].closest_neighbor_distance = distance
            self.drone_array[i].closest_neighbor_position = pos

    def draw_closest_neighbor(self):
        for i in range(0, SWARM_SIZE):
            first_drone_pos  = (self.drone_array[i].x, self.drone_array[i].y)
            second_drone_pos = self.drone_array[i].closest_neighbor_position
            pygame.draw.line(self.display_surf, RED, first_drone_pos, second_drone_pos, 1)

    def move_drones(self):
        for i in range(0, SWARM_SIZE):
            self.drone_array[i].calculate_and_move(self.circle_present, self.circler, self.circlex, self.circley)

    def draw_circle(self):
        circle = None
        self.circle_present = True
        self.circler = random.randint(3, HEIGHT / 2 - 1)
        self.circlex = random.randint(self.circler, WIDTH - self.circler)
        self.circley = random.randint(self.circler, HEIGHT - self.circler)
        circle = Circle(self, self.circlex, self.circley, self.circler)

    def on_loop(self):
        self.calculate_drone_closeness()
        self.move_drones()

    def draw_background(self):
        self.display_surf.fill(BACKGROUND_COLOR)

    def on_render(self):
        self.draw_background()
        self.shape_sprites.update()
        self.shape_sprites.draw(self.display_surf)
        self.drone_sprites.update()
        self.drone_sprites.draw(self.display_surf)
        if(DRONE_DEBUG == True):
            self.draw_closest_neighbor()
        pygame.display.update()

    def on_cleanup(self):
        pygame.quit()

    def load_data(self):
        pass

    def on_execute(self):
        #Game loop
        if(self.on_init() == False):
            self.running = False
        while(self.running):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
            self.FramePerSec.tick(self.FPS)
        self.on_cleanup()

if __name__ == "__main__":
    theApp = Game()
    theApp.on_execute()