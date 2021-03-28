import numpy as np
import math
import random
import json
import matplotlib.pyplot as plt
import matplotlib.animation as anim

TIME = "Time Step"
FREE_ENERGY = "Free Energy"

# np.roll(self.phi_0, 1, axis=0) #top
# np.roll(self.phi_0, 1, axis=1) #left
# np.roll(self.phi_0, -1, axis=1) #right
# np.roll(self.phi_0, -1, axis=0) #bottom

class Simulation():

    def __init__(self):
        self.json_object = {}
        self.json_object[TIME] = []
        self.json_object[FREE_ENERGY] = []

    def Start(self):
        self.size = 100
        self.loops = 100000
        self.delta_t = 1
        self.delta_x = 1
        self.M = 0.1
        self.a = 0.1
        self.k = 0.1
        self.c_1 = self.k / self.delta_x**2
        self.c_2 = self.M * self.delta_t / self.delta_x**2
        self.length = self.size ** 2
        # self.c_1 = self.k / self.delta_x ** 2 * self.a
        # self.c_2 = self.M * self.delta_t / self.delta_x ** 2
        self.phi_initial = 0
        self.phi_0 = np.random.uniform(self.phi_initial-0.1,self.phi_initial+0.1, (self.size,self.size))
        self.VisualizationUpdate()
        #self.DataCollectionUpdate()

    def Update(self):
        mu = self.a * self.phi_0 * (-1 + self.phi_0**2) \
                    - self.c_1 * (np.roll(self.phi_0, -1, axis=1) + np.roll(self.phi_0, 1, axis=1) \
                    + np.roll(self.phi_0, 1, axis=0) + np.roll(self.phi_0, -1, axis=0) - 4 * self.phi_0)

        self.phi_0 += self.c_2 \
                * (np.roll(mu, -1, axis=1) + np.roll(mu, 1, axis=1) \
                + np.roll(mu, 1, axis=0) + np.roll(mu, -1, axis=0) - 4 * mu)


##########################################################################################################

    def VisualizationUpdate(self):
        figure, self.data_points = Simulation.CreateFigure(self.size)
        self.animation = anim.FuncAnimation(figure, func=self.Animate, frames=self.LoopFunction, interval=1, blit=False, repeat=False)
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
        plt.colorbar(data_points)
        return figure, data_points

    def LoopFunction(self):
        for i in range(self.loops):
            if i % 100 == 0:
                print(i)
            self.Update()
            if i % 100 == 0:
                yield self.phi_0

##########################################################################################################

    def DataCollectionUpdate(self):
        self.json_object[TIME] = np.arange(0,self.loops,100).tolist()
        for i in range(self.loops):
            self.Update()
            if i % 100 == 0:
                print(i)
                free_energy_density = self.a * self.phi_0**2 / 2 * (-1 + self.phi_0**2 / 2) \
                     + self.c_1/8 * ((np.roll(self.phi_0, -1, axis=1)-np.roll(self.phi_0, 1, axis=1))**2 \
                                    + (np.roll(self.phi_0, 1, axis=0)-np.roll(self.phi_0, -1, axis=0))**2)
            
                free_energy = np.sum(free_energy_density)
                self.json_object[FREE_ENERGY].append(free_energy)
        self.SaveData("data.jsonc")
        self.PlotData("data.jsonc")

    def SaveData(self, file_path):
        with open(file_path, 'w') as outfile:
            json.dump(self.json_object, outfile)

    # Plots gathered data by reading json file
    def PlotData(self, file_path):
        with open(file_path) as json_file:
            # Load the json object from the file
            j = json.load(json_file)
            times = j.get(TIME)

            Simulation.FormatPlot(plt.plot(times, j.get(FREE_ENERGY)), "Free Energy, phi_0 = 0", "Time", "Free Energy")
            print("Finished")

    # Function that allows for many plots to be made in less space
    @staticmethod
    def FormatPlot(plot, title, x_axis, y_axis):
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()

##########################################################################################################

sim = Simulation()

sim.Start()