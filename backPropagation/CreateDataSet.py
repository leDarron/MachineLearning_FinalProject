import pygame
import math
import random
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LinearRing
import json


pygame.init()

HEIGHT, WIDTH = 12, 9
IMAGE_SIZE = 80
FPS = 60
BACK_COLOR = (200, 220, 240)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
JUMP_COULDOWN = 6
JUMP_VALUE = -7
INITIAL_BIRD_SPEED = IMAGE_SIZE / 16
BIRD_ACCELERATION = IMAGE_SIZE / 3200
DIFFICULTY_BASE = 0
DIFFICULTY_INTERVAL = 20
DIFFICULTY_INCR = IMAGE_SIZE / 6
SPIKE_SPEED = 0.05
GRAVITY = 0.4

POPULATION = 50
PUNISHMENT_H = 10
PUNISHMENT_V = 20
MAX_GENERATION = 50
TRESHOLD_JUMP = 0.7

BLUE_JUMP = pygame.transform.scale(pygame.image.load("../ressources/blue_jump.png"), (IMAGE_SIZE, IMAGE_SIZE))
BLUE_FALL = pygame.transform.scale(pygame.image.load("../ressources/blue_fall.png"), (IMAGE_SIZE, IMAGE_SIZE))
RED_JUMP = pygame.transform.scale(pygame.image.load("../ressources/red_jump.png"), (IMAGE_SIZE, IMAGE_SIZE))
RED_FALL = pygame.transform.scale(pygame.image.load("../ressources/red_fall.png"), (IMAGE_SIZE, IMAGE_SIZE))
SPIKE_LEFT = pygame.transform.scale(pygame.image.load("../ressources/spike_left.png"), (IMAGE_SIZE, IMAGE_SIZE))
SPIKE_RIGHT = pygame.transform.scale(pygame.image.load("../ressources/spike_right.png"), (IMAGE_SIZE, IMAGE_SIZE))
SPIKE_DOWN = pygame.transform.scale(pygame.image.load("../ressources/spike_down.png"), (IMAGE_SIZE, IMAGE_SIZE))
SPIKE_UP = pygame.transform.scale(pygame.image.load("../ressources/spike_up.png"), (IMAGE_SIZE, IMAGE_SIZE))
WALL = pygame.transform.scale(pygame.image.load("../ressources/wall.png"), (IMAGE_SIZE, IMAGE_SIZE))
ADN = pygame.transform.scale(pygame.image.load("../ressources/adn.png"), (IMAGE_SIZE, IMAGE_SIZE))
BIRDS = pygame.transform.scale(pygame.image.load("../ressources/birds.png"), (IMAGE_SIZE, IMAGE_SIZE))

GAME_WIDTH = WIDTH * IMAGE_SIZE
GAME_HEIGHT = HEIGHT * IMAGE_SIZE
SCREEN_WIDTH = GAME_WIDTH
SCREEN_HEIGHT = GAME_HEIGHT
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Don't Touch the Spikes")
SCORE_FONT = pygame.font.Font("../ressources/font.otf", int(IMAGE_SIZE * 2.5))
TEXT_FONT = pygame.font.Font("../ressources/font.otf", int(IMAGE_SIZE))


class Bird():
    def __init__(self):
        self.y = HEIGHT * IMAGE_SIZE // 2 - IMAGE_SIZE // 2
        self.vspeed = JUMP_VALUE
        self.jumping = 0
        self.couldown = 0
        self.fitness = 0
    
    def get_input(self, distance_to_wall, hspeed, up, down, predict_spike_offset):
        net_input = [self.y / GAME_HEIGHT,
                     distance_to_wall / GAME_WIDTH,
                     abs(hspeed) / 10,
                     (self.y - up - predict_spike_offset) / GAME_HEIGHT,
                     (self.y - down - predict_spike_offset) / GAME_HEIGHT]
        return net_input
        
    def jump(self):
        self.jumping = 20
        self.vspeed = JUMP_VALUE
        self.couldown = JUMP_COULDOWN

    def move(self):
        self.couldown -= 1
        self.y += self.vspeed
        self.vspeed += GRAVITY
    
    def draw(self, birds_x, hspeed, last):
        if self.jumping > 0:
            self.jumping -= 1
            if last:
                _sprite_bird = RED_JUMP
            else:
                _sprite_bird = BLUE_JUMP
        else:
            if last:
                _sprite_bird = RED_FALL
            else:
                _sprite_bird = BLUE_FALL
        if hspeed < 0:
            _sprite_bird = pygame.transform.flip(_sprite_bird, True, False)
        SCREEN.blit(_sprite_bird, (birds_x, self.y))



class Spike():
    def __init__(self, x, y, sprite):
        self.decal_speed = -IMAGE_SIZE / 32
        self.decal = IMAGE_SIZE / 2
        self.sprite = sprite
        self.position = (x, y)
        if sprite == SPIKE_LEFT:
            self.hitbox = [(x, y), (x, y + IMAGE_SIZE), (x + IMAGE_SIZE / 2, y + IMAGE_SIZE / 2)]
        if sprite == SPIKE_RIGHT:
            self.hitbox = [(x + IMAGE_SIZE, y), (x + IMAGE_SIZE, y + IMAGE_SIZE), (x + IMAGE_SIZE / 2, y + IMAGE_SIZE / 2)]
        if sprite == SPIKE_UP:
            self.hitbox = [(x, y), (x + IMAGE_SIZE, y), (x + IMAGE_SIZE / 2, y + IMAGE_SIZE / 2)]
        if sprite == SPIKE_DOWN:
            self.hitbox = [(x, y + IMAGE_SIZE), (x + IMAGE_SIZE, y + IMAGE_SIZE), (x + IMAGE_SIZE / 2, y + IMAGE_SIZE / 2)]

    def draw(self, spike_offset):
        self.decal += self.decal_speed
        if self.decal <= IMAGE_SIZE / 16:
            self.decal = IMAGE_SIZE / 16
            self.decal_speed = 0
        if self.decal > IMAGE_SIZE / 2 - IMAGE_SIZE / 16:
            return False
        x, y = self.position
        if self.sprite == SPIKE_RIGHT:
            x += self.decal
        else:
            x -= self.decal
        SCREEN.blit(self.sprite, (x, y + spike_offset))
        return True


def create_spikes(spikes_list, score):
    for spike in spikes_list:
        if spike.sprite in [SPIKE_LEFT, SPIKE_RIGHT]:
            spike.decal_speed = IMAGE_SIZE / 32

    available_yspikes = list(range(2, HEIGHT - 2))
    nb_max = len(available_yspikes)
    slots = [0] * nb_max
    random.shuffle(available_yspikes)
    
    nb_spikes = int(2 + (nb_max - 4) * (score / 70))
    nb_spikes = max(2, min(nb_spikes, nb_max - 2))
    
    for _ in range(nb_spikes):
        yspike = available_yspikes.pop()
        slots[yspike - 2] = 1
        yspike *= IMAGE_SIZE
        if score % 2 == 1:
            spike = Spike(IMAGE_SIZE - IMAGE_SIZE / 16, yspike, SPIKE_LEFT)
        else:
            spike = Spike(WIDTH * IMAGE_SIZE - (IMAGE_SIZE * 2) + IMAGE_SIZE / 16, yspike, SPIKE_RIGHT)
        spikes_list.append(spike)

    count = 0
    max_count = 0
    end_index = 0
    size = len(slots)
    for index, element in enumerate(slots):
        if element == 0:
            count += 1
            if count > max_count:
                max_count = count
                end_index = index
            elif count == max_count:
                pos = index - count / 2
                best_post = end_index - max_count / 2
                if abs(pos - size / 2) < abs(best_post - size / 2):
                    max_count = count
                    end_index = index
        else:
            count = 0
    start_index = end_index - max_count + 1

    up = (start_index + 2) * IMAGE_SIZE
    down = (end_index + 2) * IMAGE_SIZE

    return up, down, spikes_list


def collide_h(x, y, spikes_list, spike_offset):
    if x > IMAGE_SIZE * 2 and x < GAME_WIDTH - GAME_WIDTH * 3:
        return False
    for spike in spikes_list:
        if spike.decal != IMAGE_SIZE / 16:
            continue
        a, b, c = spike.hitbox
        spike_points = [a, b, c]
        triangle_points = []
        for point in spike_points:
            spike_x, spike_y = point
            triangle_points.append((spike_x, spike_y + spike_offset))
        triangle = Polygon(triangle_points)
        circle = Point((x + IMAGE_SIZE / 2, y + IMAGE_SIZE / 2)).buffer(IMAGE_SIZE / 2)
        if triangle.intersects(circle):
            return True
        else:
            triangle_ring = LinearRing(triangle.exterior.coords)
            if triangle_ring.intersects(circle):
                return True
    return False

def collide_v(y):
    if y + IMAGE_SIZE / 4 < IMAGE_SIZE * 1.5 or y + IMAGE_SIZE - IMAGE_SIZE / 4 > GAME_HEIGHT - IMAGE_SIZE * 1.5:
        return True
    return False


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def draw_game(bird, spikes, score, birds_x, hspeed, spike_offset, up, down):
    SCREEN.fill(BACK_COLOR)
    pygame.draw.circle(SCREEN, WHITE, (WIDTH * IMAGE_SIZE // 2, HEIGHT * IMAGE_SIZE // 2), IMAGE_SIZE * 2.5)
    score_text = SCORE_FONT.render(f"{str(int(score)).zfill(3)}", True, BACK_COLOR)
    score_rect = score_text.get_rect(center=(WIDTH * IMAGE_SIZE // 2, HEIGHT * IMAGE_SIZE // 2 + IMAGE_SIZE * 0.4))
    SCREEN.blit(score_text, score_rect)

    bird.draw(birds_x, hspeed, 1)
    
    new_spikes = []
    for spike in spikes:
        if spike.draw(spike_offset):
            new_spikes.append(spike)
    spikes = new_spikes

    for y in range(0, HEIGHT):
        for x in range(0, WIDTH):
            if x == 0 or y == 0 or x == WIDTH - 1 or y == HEIGHT - 1:
                SCREEN.blit(WALL, (x * IMAGE_SIZE, y * IMAGE_SIZE))
    
    for x in range(0, WIDTH):
        SCREEN.blit(SPIKE_UP, (x * IMAGE_SIZE, IMAGE_SIZE - IMAGE_SIZE / 16))
        SCREEN.blit(SPIKE_DOWN, (x * IMAGE_SIZE, GAME_HEIGHT - 2 * IMAGE_SIZE + IMAGE_SIZE / 16))

    pygame.draw.line(SCREEN, (50, 150, 50), (0 + IMAGE_SIZE, up), (GAME_WIDTH - IMAGE_SIZE, up), 2)
    pygame.draw.line(SCREEN, (50, 150, 50), (0 + IMAGE_SIZE, down + IMAGE_SIZE), (GAME_WIDTH - IMAGE_SIZE, down + IMAGE_SIZE), 2)

    
    pygame.display.update()
    
    return spikes



######################################################################################
## ------------------------------------- MAIN ------------------------------------- ##
######################################################################################



def main_loop(data_set):
    score = 0
    clock = pygame.time.Clock()
    
    bird = Bird()
    spikes_list = []
    
    run = True
    time = 0
    spike_offset = 0
    predict_spike_offset = 0
    difficulty = DIFFICULTY_BASE
    up = GAME_HEIGHT / 2
    down = GAME_HEIGHT / 2
    birds_x = WIDTH * IMAGE_SIZE // 2 - IMAGE_SIZE // 2
    hspeed = INITIAL_BIRD_SPEED
    distance_to_wall = IMAGE_SIZE * (WIDTH - 1)
    game_time = 0
    
    while run is True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird.jump()
                    data_set.append((bird.get_input(distance_to_wall, hspeed, up, down, predict_spike_offset), 1))
        clock.tick(FPS)
        time += SPIKE_SPEED
        spike_offset = math.cos(time) * difficulty
        
        bird.move()
        bird_failed_h = True if collide_h(birds_x, bird.y, spikes_list, spike_offset) is True else False
        bird_failed_v = True if collide_v(bird.y) is True else False
        if bird_failed_h or bird_failed_v:
            run = False
        
        birds_x += hspeed
        if birds_x < IMAGE_SIZE or birds_x > (WIDTH - 2) * IMAGE_SIZE:
            if hspeed > 0:
                hspeed += BIRD_ACCELERATION
            else:
                hspeed -= BIRD_ACCELERATION
            hspeed *= -1
            score += 1
            up, down, spikes_list = create_spikes(spikes_list, score)
            if score % DIFFICULTY_INTERVAL == 0:
                difficulty += DIFFICULTY_INCR

        if hspeed > 0:
            nearest_wall_x = (WIDTH - 2) * IMAGE_SIZE
        else:
            nearest_wall_x = IMAGE_SIZE
        distance_to_wall = nearest_wall_x - birds_x
        
        predict_time = time + SPIKE_SPEED * (distance_to_wall / hspeed)
        predict_spike_offset = math.cos(predict_time) * difficulty

        draw_game(bird, spikes_list, score, birds_x, hspeed, spike_offset, up, down)
        
        game_time += 1
        if game_time % 50 == 0 and bird.jumping != 19:
            data_set.append((bird.get_input(distance_to_wall, hspeed, up, down, predict_spike_offset), 0))
    
    return True
    
    

def main():
    data_set = []
    running = True
    while running:
        running = main_loop(data_set)
    
    data = json.dumps(data_set)
    with open("data_set.txt", "w") as file:
        file.write(data)
        
main()