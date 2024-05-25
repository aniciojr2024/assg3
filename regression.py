import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    # Create a linear regression model with the given input and output sizes
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)  # Forward pass: compute the predicted values
    loss = loss_fn(pred, y)  # Calculate the loss using the loss function

    # Backpropagation
    optimizer.zero_grad()  # Clear the gradients of all optimized tensors
    loss.backward()  # Backpropagation: compute gradients
    optimizer.step()  # Update model parameters based on the computed gradients
    return loss


def fit_regression_model(X, y):
    # Define the learning rate and number of epochs for training
    learning_rate = 0.001  
    num_epochs = 10000   
    input_features = X.shape[1]  # Number of input features
    output_features = y.shape[1]  # Number of output features
    
    # Create the linear regression model
    model = create_linear_regression_model(input_features, output_features)
    
    # Define the loss function (mean squared error loss)
    loss_fn = nn.MSELoss()

    # Define the optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")  # Initialize previous loss for early stopping
    tolerance = 1e-8  # Define the tolerance for early stopping

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        
        # Print the loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

        # Check if the loss has not changed significantly
        if abs(previous_loss - loss.item()) < tolerance:
            print(f'Training stopped at epoch {epoch} with loss {loss.item()}')
            break
        
        previous_loss = loss.item()  # Update previous loss
    
    return model, loss
