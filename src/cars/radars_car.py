# TODO - Change this file: it is using an older type of car based on decision trees

from global_vars import *
from cars.base_car import *
from decision_trees import *

class AntecipateCar(DTCar):
    IMG = PURPLE_CAR
    START_POS = (150, 200)

    # Action Nodes
    move_forward_action = Action("move_forward")  # Action already defined in AbstractCar
    turn_left_action = Action("turn_left")
    turn_right_action = Action("turn_right")

    # Decision Nodes
    top_bow_right = Boolean("top_bow_right", turn_left_action, move_forward_action)
    top_bow_left = Boolean("top_bow_left", turn_right_action, top_bow_right)
    top_bow_middle = Boolean("top_bow_middle", top_bow_left, move_forward_action)

    # Decision Tree
    DT = top_bow_middle

    def update_sensors(self):
        self.sensors["top_bow_left"] = TRACK_MASK.get_at(self.bullets["top_bow_left"]) != WHITE
        self.sensors["top_bow_middle"] = TRACK_MASK.get_at(self.bullets["top_bow_middle"]) != WHITE
        self.sensors["top_bow_right"] = TRACK_MASK.get_at(self.bullets["top_bow_right"]) != WHITE
        # self.sensors["bottom"] = TRACK_MASK.get_at(self.bullets["bottom"]) != WHITE

    def step(self, verbose=False):
        self.update_sensors()
        action = self.DT.decide(self.sensors)
        eval('self.' + action + '()')
        if verbose:
            print(f"\tPurple: {action} with vel={self.vel};acc={self.acceleration};angle={self.angle}")

    # Actions

    def turn_left(self):
        self.rotate(left=True)
        self.move_forward_while_turning()

    def turn_right(self):
        self.rotate(right=True)
        self.move_forward_while_turning()

    # Utils for actions

    def move_forward_while_turning(self):
        self.vel = min(0.75 * self.vel + 0.25 * self.acceleration, self.max_vel)
        self.move()