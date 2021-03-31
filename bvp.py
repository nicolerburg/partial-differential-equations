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
                    "J": [self.Jacobi], \
                    "G": [self.GaussSeidel]
                }

        self.json_object = {}
        self.json_object[TIME] = []
        self.json_object[FREE_ENERGY] = []

    def Start(self):
        self.size = 50
        self.loops = 500000

        self.size = Simulation.ParseInput("Specify the size of the lattice: ", int)
        self.tolerance = Simulation.ParseInput("Enter a tolerance value: ", float)
        self.method_choice = Simulation.ParseChoices("Use Jacobi or Gauss-Seidel? [J/G]: ", ["J", "G"])

        self.charges = np.zeros((self.size,self.size, self.size))
        self.charges[int(self.size/2)][int(self.size/2)][int(self.size/2)] = 1
        self.potentials = np.zeros((self.size,self.size, self.size))

        self.DataCollectionUpdate()

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

    def Jacobi(self):
        before_grid = np.copy(self.potentials)

        potential_center = self.potentials[1:-1, 1:-1, 1:-1]
        charges_center = self.charges[1:-1, 1:-1, 1:-1]

        potential_top = self.potentials[0:-2, 1:-1, 1:-1]
        potential_bottom = self.potentials[2:, 1:-1,  1:-1]
        potential_left = self.potentials[1:-1, 0:-2,  1:-1]
        potential_right = self.potentials[1:-1, 2:, 1:-1]
        potential_front = self.potentials[1:-1, 1:-1, 0:-2]
        potential_back = self.potentials[1:-1, 1:-1, 2:]

        potential_center = 1/6 * (potential_top + potential_bottom + potential_left + potential_right + potential_front + potential_back + charges_center)

        self.potentials = np.pad(potential_center, 1)

        self.sum_difference = np.sum(np.abs(self.potentials - before_grid))

    def GaussSeidel(self):
        before_grid = np.copy(self.potentials)
        for i in range(1, self.size-1):
            for j in range(1, self.size-1):
                for k in range(1, self.size-1):
                    self.potentials[i, j, k] = 1/6 * (self.potentials[i+1, j, k] + self.potentials[i-1, j, k] \
                                              + self.potentials[i, j+1, k] + self.potentials[i, j-1, k] \
                                              + self.potentials[i, j, k+1] + self.potentials[i, j, k-1] + self.charges[i, j, k])
        self.sum_difference = np.sum(np.abs(self.potentials - before_grid))

#################### Function which control the data collection ###############################################################

    def DataCollectionUpdate(self):
        self.file_path = input("Creat file name for data file. (Do not include .json)") + ".jsonc"

        for i in range(self.loops):
            if i % 100 == 0:
                print(i)
            self.choices[self.method_choice][0]()
            if self.sum_difference < 0.001:
                break

        plt.imshow(self.potentials[int(self.size/2)])
        plt.colorbar()
        plt.show()

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

sim = Simulation()

sim.Start()