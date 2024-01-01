import pygame
import math
import random
import neat
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import LinearRing



pygame.init()

HEIGHT, WIDTH = 12, 9
IMAGE_SIZE = 80
FPS = 144
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
SCREEN_WIDTH = GAME_WIDTH * 2
SCREEN_HEIGHT = GAME_HEIGHT
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
SURFACE = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
pygame.display.set_caption("Don't Touch the Spikes")
SCORE_FONT = pygame.font.Font("../ressources/font.otf", int(IMAGE_SIZE * 2.5))
TEXT_FONT = pygame.font.Font("../ressources/font.otf", int(IMAGE_SIZE))



class Bird():
    def __init__(self, genome, config):
        self.y = HEIGHT * IMAGE_SIZE // 2 - IMAGE_SIZE // 2
        self.vspeed = JUMP_VALUE
        self.jumping = 0
        self.couldown = 0
        self.genome = genome
        self.genome.fitness = 0
        self.network = neat.nn.FeedForwardNetwork.create(self.genome, config)

    def predict(self, distance_to_wall, hspeed, up, down, predict_spike_offset):
        net_input = (self.y / GAME_HEIGHT,
                    distance_to_wall / GAME_WIDTH,
                    abs(hspeed) / 10,
                    (self.y - up - predict_spike_offset) / GAME_HEIGHT,
                    (self.y - down - predict_spike_offset) / GAME_HEIGHT)
        output = self.network.activate(net_input)
        if output[0] > TRESHOLD_JUMP:
            self.jump()
        

    def jump(self):
        if self.couldown <= 0:
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


def draw_network(model: neat.nn.FeedForwardNetwork, generation, population):
    pygame.draw.rect(SURFACE, BLACK, (0, 0, GAME_WIDTH, GAME_HEIGHT))
    
    SURFACE.blit(ADN, (IMAGE_SIZE / 4, IMAGE_SIZE / 4))
    generation_text = TEXT_FONT.render(f"{generation}", True, WHITE)
    generation_rect = generation_text.get_rect(topleft=(IMAGE_SIZE / 4 + IMAGE_SIZE * 1.25, IMAGE_SIZE * 0.4))
    SURFACE.blit(generation_text, generation_rect)
    
    SURFACE.blit(BIRDS, (GAME_WIDTH - IMAGE_SIZE * 1.25, IMAGE_SIZE / 4))
    Population_text = TEXT_FONT.render(f"{population}", True, WHITE)
    Population_rect = Population_text.get_rect(topright=(GAME_WIDTH - IMAGE_SIZE * 1.5, IMAGE_SIZE * 0.4))
    SURFACE.blit(Population_text, Population_rect)
    
    layers = [[], [], []]

    for node_info in model.node_evals:
        node_id, _, _, _, _, links = node_info
        if node_id == 0:
            layers[2].append((node_id, links, 0))
        else:
            layers[1].append((node_id, links, 0))

    for node_id, value in model.values.items():
        if node_id < 0:
            layers[0].append((node_id, [], value))
        else:
            for layer in layers[1:]:
                for i, node in enumerate(layer):
                    node_id_in_layer, links, _ = node
                    if node_id_in_layer == node_id:
                        layer[i] = (node_id, links, value)
    
    if layers[1] == []:
        del layers[1]

    nb_neuron = 0
    for layer in layers:
        if len(layer) > nb_neuron:
            nb_neuron = len(layer)
    neuron_radius = GAME_HEIGHT / (nb_neuron * 4)
    layer_distance = GAME_WIDTH / len(layers)
    neuron_distance = neuron_radius * 2.5

    for layer_index, layer in enumerate(layers):
        x = layer_index * layer_distance + layer_distance / 2
        total_neurons_in_layer = len(layer)
        total_height = total_neurons_in_layer * neuron_distance
        vertical_offset = (GAME_HEIGHT - total_height) / 2 + neuron_distance / 2
        for neuron_index, neuron in enumerate(layer):
            y = neuron_index * neuron_distance + vertical_offset
            if layer_index > 0:
                for prev_neuron_index, prev_neuron in enumerate(layers[layer_index - 1]):
                    prev_y = prev_neuron_index * neuron_distance + vertical_offset
                    connections = [link[1] for link in neuron[1] if link[0] == prev_neuron[0]]
                    if connections:
                        weight = connections[0]
                        line_thickness = int(abs(weight) * 5) + 1
                        pygame.draw.line(SURFACE, (255, 255, 255, 255), (x, y), (x - layer_distance, prev_y + (old_vertical_offset - vertical_offset)), line_thickness)
        old_vertical_offset = vertical_offset
    
    for layer_index, layer in enumerate(layers):
        x = layer_index * layer_distance + layer_distance / 2
        total_neurons_in_layer = len(layer)
        total_height = total_neurons_in_layer * neuron_distance
        vertical_offset = (GAME_HEIGHT - total_height) / 2  + neuron_distance / 2
        for neuron_index, neuron in enumerate(layer):
            y = neuron_index * neuron_distance + vertical_offset
            pygame.draw.circle(SURFACE, (255, 255, 255, 255), (x, y), neuron_radius)
            pygame.draw.circle(SURFACE, pygame.Color(255, 0, 0, int(sigmoid(neuron[2]) * 255)), (x, y), neuron_radius)
            if neuron[0] == 0 and neuron[2] > TRESHOLD_JUMP:
                pygame.draw.circle(SURFACE, (255, 255, 0, 255), (x, y), neuron_radius)
        
    

def draw_game(birds, spikes, score, generation, birds_x, hspeed, spike_offset):
    SCREEN.fill(BACK_COLOR)
    pygame.draw.circle(SCREEN, WHITE, (WIDTH * IMAGE_SIZE // 2, HEIGHT * IMAGE_SIZE // 2), IMAGE_SIZE * 2.5)
    score_text = SCORE_FONT.render(f"{str(int(score)).zfill(3)}", True, BACK_COLOR)
    score_rect = score_text.get_rect(center=(WIDTH * IMAGE_SIZE // 2, HEIGHT * IMAGE_SIZE // 2 + IMAGE_SIZE * 0.4))
    SCREEN.blit(score_text, score_rect)

    for index, bird in enumerate(birds):
        bird.draw(birds_x, hspeed, index == len(birds) - 1)
    
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
    
    if len(birds) > 0:
        draw_network(birds[len(birds) - 1].network, generation, len(birds))
    
    SCREEN.blit(SURFACE, (GAME_WIDTH, 0))
    
    pygame.display.update()
    
    return spikes



######################################################################################
## ------------------------------------- MAIN ------------------------------------- ##
######################################################################################



generation = 0

def main(genomes, config):
    global generation
    generation += 1
    
    score = 0
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()
    
    birds_list = []
    spikes_list = []
    
    for _, genome in genomes:
        birds_list.append(Bird(genome, config))
    
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
    
    while run is True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        if len(birds_list) == 0:
            run = False
            break
        game_time = round((pygame.time.get_ticks() - start_time)/1000, 2)
        clock.tick(FPS)
        time += SPIKE_SPEED
        spike_offset = math.cos(time) * difficulty
        
        for index, bird in enumerate(birds_list):
            bird.move()
            bird.predict(distance_to_wall, hspeed, up, down, predict_spike_offset)
            
            bird_failed_h = True if collide_h(birds_x, bird.y, spikes_list, spike_offset) is True else False
            bird_failed_v = True if collide_v(bird.y) is True else False
            
            bird.genome.fitness = game_time + score - bird_failed_h * PUNISHMENT_H - bird_failed_v * PUNISHMENT_V
            
            if bird_failed_h or bird_failed_v:
                birds_list.pop(index)
        
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

        draw_game(birds_list, spikes_list, score, generation, birds_x, hspeed, spike_offset)



def run_NEAT(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    neat_pop = neat.population.Population(config)
    neat_pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    neat_pop.add_reporter(stats)
    neat_pop.run(main, MAX_GENERATION)
    winner = stats.best_genome()
    print('\nBest genome:\n{!s}'.format(winner))

config_file = 'config-feedforward.txt'
run_NEAT(config_file)