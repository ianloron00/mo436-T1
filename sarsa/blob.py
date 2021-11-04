
import numpy as np
from . import MOVEMENT


class Blob(object):
    def __init__(self, size):
        self.size = size - 1
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self) -> str:
        return f"{self.x}, {self.y}"

    def __sub__(self, blob) -> tuple:
        return (self.x-blob.x, self.y-blob.y)

    def __eq__(self, blob) -> bool:
        return self.x == blob.x and self.y == blob.y

    def action(self, movement: MOVEMENT):
        if movement == MOVEMENT.UP:
            self.y = self.bounders(self.y - 1)
        elif movement == MOVEMENT.DOWN:
            self.y = self.bounders(self.y + 1)
        elif movement == MOVEMENT.LEFT:
            self.x = self.bounders(self.x - 1)
        elif movement == MOVEMENT.RIGHT:
            self.x = self.bounders(self.x + 1)

    def bounders(self, value):
        if value < 0:
            return 0
        elif value > self.size:
            return self.size
        else:
            return value
