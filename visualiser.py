import pygame
from enum import Enum
import numpy as np


class Colour(Enum):
    Black = (0, 0, 0)
    Grey = (100, 100, 100)
    White = (255, 255, 255)
    Red = (168, 22, 0)
    Blue = (22, 0, 168)
    Green = (0, 168, 22)
    Cyan = (0, 168, 168)
    Yellow = (200, 200, 0)
    Magenta = (168, 0, 168)
    Orange = (200, 90, 0)
    Purple = (100, 0, 168)
    LightRed = (200, 100, 100)
    LightBlue = (100, 100, 200)
    LightGreen = (100, 200, 100)
    LightCyan = (100, 200, 200)
    LightYellow = (200, 200, 100)
    LightMagenta = (200, 100, 200)
    LightPurple = (168, 100, 255)
    LightOrange = (255, 168, 100)


colour_order = [Colour.Red, Colour.Blue, Colour.Green, Colour.Cyan, Colour.Yellow, Colour.Magenta, Colour.Orange,
                Colour.Purple, Colour.LightRed, Colour.LightBlue, Colour.LightGreen, Colour.LightCyan,
                Colour.LightYellow, Colour.LightMagenta, Colour.LightOrange, Colour.LightPurple]


class Visualiser:

    def __init__(self):
        self.HEIGHT = 800
        self.AGENT_SIZE = 5
        self.initialised = False

    def setup(self, shape):
        self.shape = shape
        pygame.init()
        self.WIDTH = self.HEIGHT * int(self.shape[0] / self.shape[1])
        self.screen = pygame.display.set_mode((self.HEIGHT, self.WIDTH))
        self.clock = pygame.time.Clock()
        self.screen.fill(Colour.Black.value)

    def clear(self):
        self.screen.fill(Colour.White.value)

    def draw_past_positions(self, past_positions):
        for i, agent_positions in enumerate(past_positions):
            colour = colour_order[i % len(colour_order)].value
            pygame.draw.lines(self.screen, colour, False, agent_positions.tolist())

    def draw_agents(self, positions, velocities):
        for i, pos in enumerate(positions):
            colour = colour_order[i % len(colour_order)].value
            radius = self.AGENT_SIZE
            pygame.draw.circle(self.screen, colour, pos.tolist(), radius, 0)
            pygame.draw.circle(self.screen, Colour.Black.value, pos.tolist(), radius, 3)

    def draw_forces(self, poss, *forces):
        for i in range(len(poss)):
            for force in forces:
                pygame.draw.line(self.screen, colour_order[i % len(colour_order)].value, poss[i].tolist(), (poss[i]+10*force[i]).tolist(), 2)
        pygame.display.flip()

    def render(self, positions, velocities, past_positions):
        if not self.initialised:
            self.setup((1, 1))
            self.initialised = True
        self.clear()
        self.draw_past_positions(past_positions)
        self.draw_agents(positions, velocities)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
