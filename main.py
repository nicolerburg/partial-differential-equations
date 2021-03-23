import numpy as np
import math
import random
import json
import matplotlib.pyplot as plt
import matplotlib.animation as anim

class Simulation():

    def Start(self):
        self.size = 50
        self.loops = 1000
        self.delta_t = 1
        self.delta_x = 1
        self.M = 0.1
        self.a = 0.1
        self.k = 0.1
        self.phi_0 = np.random.uniform(-0.1, 0.1, size=(self.size, self.size))

    def Update(self):
        mu = 42
        return self.grid[y, x] + (self.M * self.delta_t / self.delta_x**2) \
            * (mu[(y + 1) % self.size, x] + self.grid[(y - 1)  % self.size, x] \
            + self.grid[y, (x + 1)  % self.size] + self.grid[y, (x - 1)  % self.size])

    def VisualizationUpdate(self):
        figure, self.data_points = Simulation.CreateFigure(self.size)
        self.animation = anim.FuncAnimation(figure, func=self.Animate, frames=self.LoopFunction, interval=10, blit=False, repeat=False)
        plt.show()

    def Animate(self, grid):
        self.data_points.set_data(grid)
        return self.data_points

    @staticmethod
    def CreateFigure(size):
        figure, axes = plt.subplots()
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        data_points = axes.imshow(np.zeros((size, size)), vmin=-1, vmax=1)
        return figure, data_points

    def LoopFunction(self):
        for _ in range(self.loops):

            self.Update()
            yield self.phi


sim = Simulation()

sim.Start()