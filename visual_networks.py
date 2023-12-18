from OpenGL.GL import *
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

from colors import *

class Neuron:
    def __init__(self, x, y, radius=0.05):
        self.x = x
        self.y = y
        self.radius = radius
        self.value = random.random()
        #self.color = random_color()  # Random red component
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
    def __init__(self, neuron_count, layer_index, total_layers, radius=0.05):
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
            self.neurons.append(Neuron(self.get_layer_x_position(), y, radius))


    def get_layer_x_position(self):
        # Horizontal position based on layer index
        layer_gap = 2.0 / (self.total_layers + 1)
        return -1 + layer_gap + (self.layer_index * layer_gap)

    def draw(self):
        for neuron in self.neurons:
            neuron.draw()

class Connection:

    def __init__(self, start_neuron, end_neuron, width=0.5):
        self.start_neuron = start_neuron
        self.end_neuron = end_neuron
        self.width = 0.1

        self.color = random_color()  # Random red component

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
    def __init__(self, layer_structure):
        self.layers = []
        self.connections = []  # List to store connections

        # Create layers
        total_layers = len(layer_structure)
        limiting_factor = max(total_layers, max(layer_structure))

        for i, neuron_count in enumerate(layer_structure):
            self.layers.append(Layer(neuron_count, i, total_layers, 0.8/limiting_factor))

        # Create connections between layers
        for i in range(len(self.layers) - 1):
            for neuron in self.layers[i].neurons:
                for next_neuron in self.layers[i + 1].neurons:
                    self.connections.append(Connection(neuron, next_neuron))

    def draw(self):
        # Draw connections first
        for connection in self.connections:
            connection.draw()

        # Then draw layers
        for layer in self.layers:
            layer.draw()


# Function to extract model parameters
def extract_params(model):
    params_dict = {}
    for name, param in model.named_parameters():
        params_dict[name] = param.data.cpu().numpy()
    return params_dict

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

# Function to visualize the whole model
def visualize_model(model):
    params_dict = extract_params(model)
    visualize_params(params_dict)