import numpy as np
from assignment_2.layers.conv2d import Conv2D
from assignment_2.layers.relu import ReLU, LeakyReLU
from assignment_2.loss.mse import mse, mse_prime

class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_grad = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_grad):
        self.loss = loss
        self.loss_grad = loss_grad

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, learning_rate, verbose=True):
        for epoch in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # Forward
                output = x
                for layer in self.layers:
                    output = layer.forward(output)

                # Loss
                error += self.loss(y, output)

                # Backward
                grad = self.loss_grad(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Error: {error/len(x_train):.4f}")

# Dummy dataset
x_train = [np.random.rand(3, 32, 32)] * 10  # 10 samples of 3x32x32
y_train = [np.array([1, 0, 0])] * 10        # Dummy one-hot labels

# Build model
model = Model()
# model.add(...) ← We will add layers here later
model.use(mse, mse_prime)
# model.train(x_train, y_train, epochs=10, learning_rate=0.01)
