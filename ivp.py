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
        self.choices = { \
                    "D": [self.DataCollectionUpdate], \
                    "V": [self.VisualizationUpdate] \
                }

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

        self.size = Simulation.ParseInput("Specify the size of the lattice: ", int)
        self.phi_initial = Simulation.ParseInput("Enter a phi_0 value: ", float)
        method_choice = Simulation.ParseChoices("Run Visualisation or Data Collection? [V/D]: ", ["V", "D"])

        self.phi_0 = np.random.uniform(self.phi_initial-0.1,self.phi_initial+0.1, (self.size,self.size))

        self.choices[method_choice][0]()

    # Functions which parse input
    @staticmethod
    def ParseInput(prompt, type):
        try:
            user_input = type(input(prompt))
            if user_input <= 1 or user_input >= -1:
                return user_input
            else:
                print("Please enter a between 1 and -1.")
        except:
            pass
        return Simulation.ParseInput(prompt, type)

    @staticmethod
    def ParseChoices(prompt, options):
        user_input = input(prompt)
        if user_input.capitalize() in options:
            return user_input.capitalize()
        else:
            return Simulation.ParseChoices(prompt, options)


    def Update(self):
        mu = self.a * self.phi_0 * (-1 + self.phi_0**2) \
                    - self.c_1 * (np.roll(self.phi_0, -1, axis=1) + np.roll(self.phi_0, 1, axis=1) \
                    + np.roll(self.phi_0, 1, axis=0) + np.roll(self.phi_0, -1, axis=0) - 4 * self.phi_0)

        self.phi_0 += self.c_2 \
                * (np.roll(mu, -1, axis=1) + np.roll(mu, 1, axis=1) \
                + np.roll(mu, 1, axis=0) + np.roll(mu, -1, axis=0) - 4 * mu)


################### Function which control the live visualization #########################################################################

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

#################### Function which control the data collection ###############################################################

    def DataCollectionUpdate(self):
        self.file_path = input("Creat file name for data file. (Do not include .json)") + ".jsonc"
        self.json_object[TIME] = np.arange(0,self.loops,100).tolist()
        for i in range(self.loops):
            self.Update()
            if i % 100 == 0:
                print(i)
                free_energy_density = self.CalculateFreeEnergy()
            
                free_energy = np.sum(free_energy_density)
                self.json_object[FREE_ENERGY].append(free_energy)
        self.SaveData(self.file_path)
        self.PlotData(self.file_path)

    def SaveData(self, file_path):
        with open(file_path, 'w') as outfile:
            json.dump(self.json_object, outfile)

    # Plots gathered data by reading json file
    def PlotData(self, file_path):
        with open(file_path) as json_file:
            # Load the json object from the file
            j = json.load(json_file)
            times = j.get(TIME)

            Simulation.FormatPlot(plt.plot(times, j.get(FREE_ENERGY)), "Free Energy, phi_0 = 0.5", "Time", "Free Energy")
            print("Finished")

    # Function that allows for many plots to be made in less space
    @staticmethod
    def FormatPlot(plot, title, x_axis, y_axis):
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()

##########################################################################################################

    def CalculateFreeEnergy(self):
        return self.a * self.phi_0**2 / 2 * (-1 + self.phi_0**2 / 2) \
            + self.c_1/8 * ((np.roll(self.phi_0, -1, axis=1)-np.roll(self.phi_0, 1, axis=1))**2 \
                        + (np.roll(self.phi_0, 1, axis=0)-np.roll(self.phi_0, -1, axis=0))**2)

sim = Simulation()

sim.Start()