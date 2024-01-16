# Create a simple linear model
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleLinearModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleLinearModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        #self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(5, 5)
        #self.dropout2 = nn.Dropout(dropout_rate)
        #self.fc3 = nn.Linear(5, 15)
        #self.fc4 = nn.Linear(15, 5)
        self.fc5 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.fc3(x)
        #x = self.fc4(x)
        x = self.fc5(x)
        return x


    def train(model, inputs, targets, learning_rate=0.01, epochs=100):
        # Define the loss function and the optimizer
        criterion = nn.L1Loss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        total_loss = 0

        for epoch in range(epochs):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, targets.unsqueeze(1))  # Ensure targets are correctly shaped

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Accumulate the loss
            total_loss += loss.item()
        # Calculate the average loss
        average_loss = total_loss / epochs
        return average_loss