import numpy as np
import math
import random
import json
import matplotlib.pyplot as plt
import matplotlib.animation as anim

OMEGA = "Omega"
LOOPS = "Loops to Convergence"
POTENTIALS = "Potential"
E_X = "Electric Field X"
E_Y = "Electric Field Y"
DISTANCE = "Distance to center"
E_MAG = "Electric Field Magnitude"
B_MAG = "Magnetic Field Magnitude"

class Simulation():

    def __init__(self):
        self.choices = { \
                    "D": [self.DataCollectionUpdate], \
                    "J": [None, self.Jacobi], \
                    "G": [self.GaussSeidelInit, self.GaussSeidel], \
                    "S": [self.SorInit, self.Sor]
                }

        self.json_object = {}

    def Start(self):
        self.size = 50
        self.loops = 500000
        self.collection_choice = "NULL"

        self.size = Simulation.ParseInput("Specify the size of the lattice: ", int)
        self.tolerance = Simulation.ParseInput("Enter a tolerance value: ", float)
        self.type = Simulation.ParseChoices("Use electric charge or magnetic wire? [C/W]: ", ["C", "W"])
        self.method_choice = Simulation.ParseChoices("Use Jacobi, Gauss-Seidel, or SOR? [J/G/S]: ", ["J", "G", "S"])
        self.file_path = input("Creat file name for data file. (Do not include .json)") + ".jsonc"

        self.center = int(self.size/2)
        self.charges = np.zeros((self.size,self.size, self.size))
        if self.type == "W":
            for k in range(self.size):
                self.charges[self.center][self.center][k] = 1
        elif self.type == "C":
            self.charges[self.center][self.center][self.center] = 1
            
        try:
            self.choices[self.method_choice][0]()
        except:
            pass

        if self.method_choice == "S":
            self.collection_choice = Simulation.ParseChoices("Run all omegas or input specific? [A/I]: ", ["A", "I"])
            if self.collection_choice == "I":
                self.omega = Simulation.ParseInput("Enter an omega value: ", float)
                self.DataCollectionUpdate()
            elif self.collection_choice == "A":
                self.SorCollection()
        else:
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

########################Functions which contain the various algorithms########################################################

    # Basic Jacobi implementation using Numpy slicing
    def Jacobi(self):
        before_grid = np.copy(self.potentials)
        potential_center = (np.sum(Simulation.ReturnSides(self.potentials), axis=0) + self.charges[1:-1, 1:-1, 1:-1]) / 6
        self.potentials = np.pad(potential_center, 1)
        return np.sum(np.abs(self.potentials - before_grid))

    # Gauss-Seidel : uses checkerboard/masking method for vectorization
    def GaussSeidelInit(self):
        starting_mask = np.indices((self.size, self.size, self.size)).sum(axis=0) % 2
        self.mask = starting_mask[1:-1, 1:-1, 1:-1]

    def GaussSeidel(self):
        before_grid = np.copy(self.potentials)
        sides = Simulation.ReturnSides(self.potentials)
        potential_center = self.potentials[1:-1, 1:-1, 1:-1]
        charges_center = self.charges[1:-1, 1:-1, 1:-1]
        potential_center[self.mask==1] = (sides[0][self.mask==1] + sides[1][self.mask==1] \
                                              + sides[2][self.mask==1] + sides[3][self.mask==1] \
                                              + sides[4][self.mask==1] + sides[5][self.mask==1] + charges_center[self.mask==1]) / 6

        potential_center[self.mask==0] = (sides[0][self.mask==0] + sides[1][self.mask==0] \
                                              + sides[2][self.mask==0] + sides[3][self.mask==0] \
                                              + sides[4][self.mask==0] + sides[5][self.mask==0] + charges_center[self.mask==0]) / 6
        self.potentials = np.pad(potential_center, 1)
        return np.sum(np.abs(self.potentials - before_grid))

    # SOR : Uses Gauss-Seidel as above with omega values that can either be specified or looped through
    def SorInit(self):
        starting_mask = np.indices((self.size, self.size, self.size)).sum(axis=0) % 2
        self.mask = starting_mask[1:-1, 1:-1, 1:-1]
        self.omega_list = np.linspace(1,2, 60, endpoint=False)


    def Sor(self):
        before_grid = np.copy(self.potentials)

        potential_center = self.potentials[1:-1, 1:-1, 1:-1]
        charges_center = self.charges[1:-1, 1:-1, 1:-1]

        sides = Simulation.ReturnSides(self.potentials)

        potential_center[self.mask==1] = self.omega * (1/6 * (sides[0][self.mask==1] + sides[1][self.mask==1] \
                                                       + sides[2][self.mask==1] + sides[3][self.mask==1] \
                                                       + sides[4][self.mask==1] + sides[5][self.mask==1] + charges_center[self.mask==1])) \
                                                + (1-self.omega) * potential_center[self.mask==1]

        potential_center[self.mask==0] = self.omega * (1/6 * (sides[0][self.mask==0] + sides[1][self.mask==0] \
                                                       + sides[2][self.mask==0] + sides[3][self.mask==0] \
                                                       + sides[4][self.mask==0] + sides[5][self.mask==0] + charges_center[self.mask==0])) \
                                                + (1-self.omega) * potential_center[self.mask==0]

        self.potentials = np.pad(potential_center, 1)

        return np.sum(np.abs(self.potentials - before_grid))

    # accepts 3d data
    @staticmethod 
    def ReturnSides(data):
        return [data[0:-2, 1:-1, 1:-1], data[2:, 1:-1,  1:-1], data[1:-1, 0:-2,  1:-1], data[1:-1, 2:, 1:-1], data[1:-1, 1:-1, 0:-2], data[1:-1, 1:-1, 2:]]

    # Returns: Negative Gradients, Magnitudes, Normalized Gradients
    def GetGradients(self, grid):
        components = np.gradient(grid)
        gradients = np.stack(components, axis=3)
        magnitudes = np.linalg.norm(gradients, axis=3)

        return gradients, magnitudes, np.nan_to_num(gradients/np.stack((magnitudes,magnitudes,magnitudes), axis=3))

#################### Function which control the data collection ###############################################################

    def SorCollection(self):
        convergence_times = np.array([])
        for i in self.omega_list:
            self.omega = i
            self.DataCollectionUpdate()
            convergence_times = np.append(convergence_times, self.convergence_point)
        print(self.omega_list)
        print(convergence_times)
        self.json_object[OMEGA] = self.omega_list.tolist()
        self.json_object[LOOPS] = convergence_times.tolist()
        self.SaveData(self.file_path)
        self.PlotData(self.file_path)


    def DataCollectionUpdate(self):
        self.potentials = np.zeros((self.size,self.size, self.size))

        for i in range(self.loops):
            if i % 100 == 0:
                print(i)
            sum_difference = self.choices[self.method_choice][1]()
            if sum_difference < self.tolerance:
                self.convergence_point = i
                break

        if self.method_choice != "S" or self.collection_choice != "A":
            plt.imshow(self.potentials[:, :, self.center])
            plt.colorbar()
            plt.show()
            field, magnitudes, normalized = self.GetGradients(self.potentials)

            distances = Simulation.GetDistanceGrid((self.size, self.size), (self.center, self.center))
            self.json_object[DISTANCE] = distances.flatten().tolist()
            self.json_object[POTENTIALS] = self.potentials[:,:,self.center].flatten().tolist()

            if self.type == "W":
                Simulation.MakeQuiver(self.size, np.negative(normalized)[:,:,self.center][:,:,1], normalized[:,:,self.center][:,:,0])
                self.json_object[B_MAG] = magnitudes[:,:,self.center].flatten().tolist()
            elif self.type == "C":
                Simulation.MakeQuiver(self.size, np.negative(normalized)[:,:,self.center][:,:,0], np.negative(normalized)[:,:,self.center][:,:,1])
                self.json_object[E_MAG] = magnitudes[:,:,self.center].flatten().tolist()


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

            if self.collection_choice != "A":
                plt.scatter(np.log(j.get(DISTANCE)), np.log(j.get(POTENTIALS)), marker="x")
                plt.title("Potentials vs Distance")
                plt.xlabel("Log(Distance to center)")
                plt.ylabel("Log(Potential)")
                plt.show()
                if self.type == "W":
                    plt.scatter(np.log(j.get(DISTANCE)), np.log(j.get(B_MAG)), marker="x")
                    plt.title("Magnetic Field vs Distance")
                    plt.xlabel("Log(Distance to center)")
                    plt.ylabel("Log(B-Field Magnitude)")
                    plt.show()
                elif self.type == "C":
                    plt.scatter(np.log(j.get(DISTANCE)), np.log(j.get(E_MAG)), marker="x")
                    plt.title("Electric Field vs Distance")
                    plt.xlabel("Log(Distance to center)")
                    plt.ylabel("Log(E-Field Magnitude)")
                    plt.show()
            else:
                omegas = j.get(OMEGA)
                Simulation.FormatPlot(plt.plot(omegas, j.get(LOOPS)), "Convergence Times vs Omega", "Omega", "Loops to converge")

            print("Finished")

    # Function that allows for many plots to be made in less space
    @staticmethod
    def FormatPlot(plot, title, x_axis, y_axis):
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()

    @staticmethod
    def MakeQuiver(size, vector_x, vector_y):
        x = np.arange(0,size)
        y = np.arange(0,size)
        X, Y = np.meshgrid(x, y)
        plt.quiver(X, Y, vector_y, vector_x)
        plt.show()

##########################################################################################################

    @staticmethod
    def GetDistanceGrid(shape, point):
        x, y = np.indices(shape)
        return np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)

    def FindDistance(self, i, j):
        return np.sqrt((self.center-i)**2 + (self.center-j)**2)


sim = Simulation()

sim.Start()