from OpenGL.GL import *
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

from colors import *
from sample_networks import *

class Neuron:
    def __init__(self, x, y, value, radius=0.05):
        self.x = x
        self.y = y
        self.radius = radius
        self.value = value
        self.color = (1,1,1,0.5)

    def draw(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(*self.color)  # White color
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(self.x, self.y)
        for angle in np.linspace(0, 2 * np.pi, 15):
            glVertex2f(self.x + self.radius * np.cos(angle), self.y + self.radius * np.sin(angle))
        glEnd()

class Layer:
    def __init__(self, type, neuron_count, layer_index, total_layers, biases, radius=0.05):
        self.neurons = []
        self.layer_index = layer_index
        self.total_layers = total_layers

        # Calculate center and vertical gap between neurons
        vertical_center = 0  # This is the vertical center, adjust if your center is different
        vertical_gap = max(0.8/(neuron_count)**2, radius*2 + 0.02)  # Adjust the gap as needed
        half_height = (neuron_count - 1) * vertical_gap / 2

        for i in range(neuron_count):
            # Vertical position centered around the vertical_center
            y = vertical_center + (vertical_gap * i) - half_height

            if type == "hidden":
                self.neurons.append(Neuron(self.get_layer_x_position(), y, biases[i], radius))
            elif type == "input":
                self.neurons.append(Neuron(self.get_layer_x_position(), y, 0, radius))
            elif type == "output":
                self.neurons.append(Neuron(self.get_layer_x_position(), y, biases[i], radius))
            else:
                print("Invalid Layer Type")

    def get_layer_x_position(self):
        # Horizontal position based on layer index
        layer_gap = 2.0 / (self.total_layers + 1)
        return -1 + layer_gap + (self.layer_index * layer_gap)

    def draw(self):
        for neuron in self.neurons:
            neuron.draw()

class Connection:

    def __init__(self, start_neuron, end_neuron, value, width=0.5):
        self.start_neuron = start_neuron
        self.end_neuron = end_neuron
        self.width = 0.1

        self.color = color(value)  # Random red component

    def draw(self):
        # Set line width
        glLineWidth(self.width)

        # Optionally enable line smoothing
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glColor4f(*self.color)  # Set the random color

        glBegin(GL_LINES)
        glVertex2f(self.start_neuron.x, self.start_neuron.y)
        glVertex2f(self.end_neuron.x, self.end_neuron.y)
        glEnd()

class Network:
    def __init__(self, model):

        self.model = model
        self.network_weights, self.network_biases = self.weights_and_biases(model)

        layer_structure = [len(layer) for layer in self.network_biases]
        self.layer_structure = [len(self.network_weights[0][0])] + layer_structure

        # Create layers
        self.total_layers = len(layer_structure)
        self.limiting_factor = max(self.total_layers, max(self.layer_structure))

        self.layers = self.update_layers()
        self.connections = self.update_connections()

    def update_layers(self):
        layers = [] # List to store layers
        for i, neuron_count in enumerate(self.layer_structure):
            if i == 0:
                layers.append(Layer("input", neuron_count, i, self.total_layers, [], 0.8/self.limiting_factor))
            else:
                layers.append(Layer("hidden", neuron_count, i, self.total_layers, self.network_biases[i-1], 0.8/self.limiting_factor))
        return layers

    def update_connections(self):
        connections = []  # List to store connections
        # Create connections between layers
        for i in range(len(self.layers) - 1):
            for j in range(len(self.layers[i].neurons)):
                neuron = self.layers[i].neurons[j]
                for k in range(len(self.layers[i + 1].neurons)):
                    next_neuron = self.layers[i + 1].neurons[k]
                    connections.append(Connection(neuron, next_neuron,self.network_weights[i][k][j], 1))

                    if i==0 and j==0 and k ==0:
                        print(self.network_weights[i][k][j])
                        

        return connections
    
    def weights_and_biases(self, model):
        params = extract_params(model)
        iterator = iter(params.items())
        weights = []
        biases = []
        for first_item, second_item in zip(iterator, iterator):
            # Process the items here
            weights.append(first_item[1])
            biases.append(second_item[1])
        return weights, biases

    
    def train(self, inputs, targets):
        self.model.train(inputs, targets)
        self.network_weights, self.network_biases = self.weights_and_biases(self.model)
        self.layers = self.update_layers()
        self.connections = self.update_connections()

    def draw(self):
        # Draw connections first
        for connection in self.connections:
            connection.draw()

        # Then draw layers
        for layer in self.layers:
            layer.draw()

def extract_model_params(model):
    layers = []
    for name, param in model.named_parameters():
        if "weight" in name:
            layers.append(param.data)
    return layers

# Function to visualize model parameters
def visualize_params(params_dict):
    for name, param in params_dict.items():
        plt.figure(figsize=(10, 5))
        if 'weight' in name:
            sns.heatmap(param, annot=True, fmt=".2f")
            plt.title(f"{name} Heatmap")
        elif 'bias' in name:
            plt.hist(param, bins=20)
            plt.title(f"{name} Histogram")
        plt.show()

# Function to extract model parameters
def extract_params(model):
    params_dict = {}
    for name, param in model.named_parameters():
        params_dict[name] = param.data.cpu().numpy()
    return params_dict

# Function to visualize the whole model
def visualize_model(model):
    params_dict = extract_params(model)
    visualize_params(params_dict)