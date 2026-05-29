from utils import *

# The grass background resized 2.5x
GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
# The track image resized 0.9x
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)

TRACK_MASK = scale_image(pygame.image.load("imgs/track-mask.png"), 0.9)

# The track border image for collision detection resized 0.9x
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK=pygame.mask.from_surface(TRACK_BORDER)

# The finish image intact
FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK=pygame.mask.from_surface(FINISH)

# Import car images resized 0.55x
RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.55)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.55)
GREY_CAR = scale_image(pygame.image.load("imgs/grey-car.png"), 0.55)
PURPLE_CAR = scale_image(pygame.image.load("imgs/purple-car.png"), 0.55)

car_width,car_height=GREEN_CAR.get_size()
### Half car-width and half car-height
HALF_WIDTH=car_width/2
HALF_HEIGHT=car_height/2
CAR_SIZE=HALF_WIDTH,HALF_HEIGHT

# Get the size of the track image
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()

# The window has the size of the track image
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# The name of the window
pygame.display.set_caption("Racing Game!")

pygame.font.init()
MAIN_FONT = pygame.font.SysFont("comicsans", 44)

FPS=60

RED = (255, 0, 0, 255)
WHITE = (255, 255, 255, 255)
YELLOW = (255, 255, 0, 255)

FINISH_POSITION=(130,250)
images=[(GRASS,(0,0)),(TRACK,(0,0)),(FINISH,FINISH_POSITION),(TRACK_BORDER,(0,0))]

# The default path for the waypoints car
PATH = [
    (161,148),(147,82),(68,93),(61,174),(64,251),(59,464),
    (293,711),(399,712),(407,537),(491,474),(591,529),(618,707),
    (731,709),(739,383),(395,334),(438,250),(706,251),(738,96),
    (287,93),(273,395),(179,399),(173,258),
    #(168, 200), (161, 148)#to cross the line
]


def draw(win,images,player_car,computer_car,game_info):
    for img,pos in images:
        win.blit(img,pos)
    level_text=MAIN_FONT.render(f'Level {game_info.level}',1,(255,255,255))
    win.blit(level_text,(10,HEIGHT-level_text.get_height()-90))
    
    time_text=MAIN_FONT.render(f'Time {game_info.get_level_time()}',1,(255,255,255))
    win.blit(time_text,(10,HEIGHT-time_text.get_height()-50))
    
    velocity_text=MAIN_FONT.render(f'Vel {round(computer_car.vel,1)} px/s',1,(255,255,255))
    win.blit(velocity_text,(10,HEIGHT-velocity_text.get_height()-10))
    
    player_car.draw(win)
    computer_car.draw(win)
    pygame.display.update()