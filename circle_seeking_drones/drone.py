import pygame
from settings import *
from math import atan
from math import sin
from math import cos
from math import pi
from math import sqrt

class Drone(pygame.sprite.Sprite):
    def __init__(self, game, x, y):
        #Drone game properties
        self.groups = game.drone_sprites
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pygame.Surface((DRONE_SIZE, DRONE_SIZE))
        self.image.fill(DRONE_COLOR)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.xspeed = DRONE_SPEED
        self.yspeed = DRONE_SPEED
        self.dvtol  = 1e-6

        #Drone calculate values
        self.closest_neighbor_position = None
        self.closest_neighbor_distance = None
        self.xmin = 0
        self.xmax = WIDTH - DRONE_SIZE
        self.ymin = 0
        self.ymax = WIDTH - DRONE_SIZE

        #Drone movement values
        self.A = 0.5
        self.B = 10000000
        self.C = 10.
        self.scaling = 0.001
        self.damping = .1

        self.previous_wall_x = 0.
        self.previous_wall_y = 0.
        self.x_wall_count   = 0
        self.y_wall_count   = 0

        self.radiustol = 10
        self.radiuscount = 0
        self.closenesstol = 10
        self.closenesscount = 0

        self.timetol = 120


    def update(self):
        self.rect.x = self.x
        self.rect.y = self.y

    def calculate_and_move(self, circle_present, circler, circlex, circley):
        if(circle_present == True):
            z = circler * circler - ((self.x - circlex) ** 2.) - ((self.y - circley) ** 2.)
            dist_from_center = sqrt(((self.x - circlex) ** 2.) + ((self.y - circley) ** 2.))
        else:
            z = 0
            dist_from_center = 0
        
        delny = self.closest_neighbor_position[1] - self.y
        delnx = self.closest_neighbor_position[0] - self.x

        closest_neighbor_angle = atan(delny / delnx)

        if(delny < 0):
            if(delnx < 0):
                closest_neighbor_angle += pi
            if(delnx > 0):
                closest_neighbor_angle += 2. * pi

        if(delny > 0):
            if(delnx < 0):
                closest_neighbor_angle += pi

        neighbor_repel_force = self.B * (1. / self.closest_neighbor_distance ** 3.)



        x_drone_der = neighbor_repel_force * cos(closest_neighbor_angle)
        y_drone_der = neighbor_repel_force * sin(closest_neighbor_angle)

        x_wall_der = -self.A * (self.x - WIDTH / 2)
        
        if((self.previous_wall_x != 0.) and (x_wall_der != 0)):
            if((self.previous_wall_x / abs(self.previous_wall_x)) != (x_wall_der / abs(x_wall_der))):
                self.x_wall_count += 1
            else:
                self.x_wall_count = 0
                self.xspeed = DRONE_SPEED

        if(self.x_wall_count >= self.timetol):
            self.xspeed *= self.damping
            print("Damping x speed")

        self.previous_wall_x = x_wall_der

        if(circle_present == False):
            x_shape_der = 0
        else:
            if(z == 0):
                x_shape_der = 0
            elif(self.x - circlex == 0):
                x_shape_der = 0
            else:
                x_shape_der = self.C * ((self.x - circlex) / abs(self.x - circlex)) * (z / abs(z)) * (dist_from_center - circler) ** 2.

        x_tot_der = (x_wall_der - x_drone_der + x_shape_der) * self.scaling
        dx = (self.xspeed * x_tot_der)

        y_closest_drone_dist = -(self.y - self.closest_neighbor_position[1])

        y_wall_der = -self.A * (self.y - HEIGHT / 2)

        if((self.previous_wall_y != 0.) and (y_wall_der != 0.)):
            if((self.previous_wall_y / abs(self.previous_wall_y)) != (y_wall_der / abs(y_wall_der))):
                self.y_wall_count += 1
            else:
                self.y_wall_count = 0
                self.yspeed = DRONE_SPEED

        if(self.y_wall_count >= self.timetol):
            self.yspeed *= self.damping
            print("Damping y speed")

            self.previous_wall_y = y_wall_der

        if(circle_present == False):
            y_shape_der = 0
        else:
            if(z == 0):
                y_shape_der = 0
            elif(self.y - circley == 0):
                y_shape_der = 0
            else:
                y_shape_der = self.C * ((self.y - circley) / abs(self.y - circley)) * (z / abs(z)) * (dist_from_center - circler) ** 2.

        y_tot_der = (y_wall_der - y_drone_der + y_shape_der) * self.scaling
        dy = (self.yspeed * y_tot_der)

        if(abs(dx) > self.xspeed):
            dx = self.xspeed * dx / abs(dx)
        if(abs(dy) > self.yspeed):
            dy = self.yspeed * dy / abs(dy)

        if(abs(dx) < self.dvtol):
            dx = 0
        if(abs(dy) < self.dvtol):
            dy = 0
        self.x += dx
        self.y += dy

        if(self.x < self.xmin):
            self.x = self.xmin
        if(self.x > self.xmax):
            self.x = self.xmax
        if(self.y < self.ymin):
            self.y = self.ymin
        if(self.y > self.ymax):
            self.y = self.ymax