import sys, os, pickle, neat, pygame
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from global_vars import PATH, images, WIN, MAIN_FONT, WHITE, FPS, FINISH_POSITION, FINISH_MASK
from cars.neat_waypoint_car import NeatWaypointCar
from game import GameInfo

pygame.init()
config_path = 'config/neat_waypoints.cfg'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

with open('results/waypoints/winners/winner_final.pkl', 'rb') as f:
    winner = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(winner, config)

# Test on reversed path
car = NeatWaypointCar(net, path=list(reversed(PATH)))
clock = pygame.time.Clock()
frames = 0
while car.alive and frames < 60*60:
    clock.tick(FPS)
    for img, pos in images:
        WIN.blit(img, pos)
    car.draw(WIN)
    pygame.display.update()
    car.step(verbose=False)
    frames += 1
print(f"Reverse test: alive={car.alive}, finish={car.finish_reached}, "
      f"wp={car.waypoints_reached}/{len(PATH)}, frames={frames}")
pygame.quit()