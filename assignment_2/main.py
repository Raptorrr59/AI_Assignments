import numpy as np
from assignment_2.dataloader import load_cifar10, one_hot
from assignment_2.layers.conv2d import Conv2D
from assignment_2.layers.relu import ReLU, LeakyReLU
from assignment_2.layers.batchnorm import BatchNorm
from assignment_2.layers.batchnorm2d import BatchNorm2D
from assignment_2.layers.dropout import Dropout
from assignment_2.layers.flatten import Flatten
from assignment_2.layers.fullyconnected import FullyConnected
from assignment_2.layers.maxpooling import MaxPool2D
from assignment_2.loss.mse import mse, mse_prime
from assignment_2.optimizers import SGD, Adam

def preprocess_data(x_train, y_train, x_test, y_test):
    """Preprocess CIFAR-10 data"""
    # Data is already normalized to [0, 1] from the dataloader
    # Convert to format suitable for our layers: (N, C, H, W)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    # Convert labels to one-hot encoding
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    
    return x_train, y_train, x_test, y_test

def create_cifar10_model():
    """Create a CNN model suitable for CIFAR-10"""
    model = Model()
    
    # First conv block: 3 -> 32 channels
    model.add(Conv2D(3, 32, 3, padding=1))  # 32x32x32
    model.add(BatchNorm2D(32))
    model.add(ReLU())
    model.add(MaxPool2D(2, 2))  # 16x16x32
    
    # Second conv block: 32 -> 64 channels
    model.add(Conv2D(32, 64, 3, padding=1))  # 16x16x64
    model.add(BatchNorm2D(64))
    model.add(ReLU())
    model.add(MaxPool2D(2, 2))  # 8x8x64
    
    # Third conv block: 64 -> 128 channels
    model.add(Conv2D(64, 128, 3, padding=1))  # 8x8x128
    model.add(BatchNorm2D(128))
    model.add(ReLU())
    model.add(MaxPool2D(2, 2))  # 4x4x128
    
    # Flatten and fully connected layers
    model.add(Flatten())  # 4*4*128 = 2048
    model.add(FullyConnected(2048, 512))
    model.add(BatchNorm(512))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(FullyConnected(512, 10))  # 10 classes
    
    return model

class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_grad = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_grad):
        self.loss = loss
        self.loss_grad = loss_grad

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def predict(self, input_data):
        # Set all layers to evaluation mode
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        
        # Set all layers back to training mode
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
        
        return output

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=32, verbose=True):
        n_samples = len(x_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Set training mode
            for layer in self.layers:
                if hasattr(layer, 'training'):
                    layer.training = True
            
            total_loss = 0
            correct_predictions = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)
                
                # Calculate loss
                batch_loss = 0
                for i in range(len(x_batch)):
                    batch_loss += self.loss(y_batch[i], output[i])
                total_loss += batch_loss
                
                # Calculate accuracy
                predictions = np.argmax(output, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                correct_predictions += np.sum(predictions == true_labels)
                
                # Backward pass
                grad = np.zeros_like(output)
                for i in range(len(x_batch)):
                    grad[i] = self.loss_grad(y_batch[i], output[i])
                
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, self.optimizer)
            
            # Calculate metrics
            avg_loss = total_loss / n_samples
            accuracy = correct_predictions / n_samples
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Validation
            if x_val is not None and y_val is not None:
                val_loss, val_accuracy = self.evaluate(x_val, y_val, batch_size)
                if verbose:
                    print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    def evaluate(self, x_test, y_test, batch_size=32):
        """Evaluate the model on test data"""
        # Set evaluation mode
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        
        n_samples = len(x_test)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        total_loss = 0
        correct_predictions = 0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            x_batch = x_test[start_idx:end_idx]
            y_batch = y_test[start_idx:end_idx]
            
            # Forward pass
            output = x_batch
            for layer in self.layers:
                output = layer.forward(output)
            
            # Calculate loss and accuracy
            for i in range(len(x_batch)):
                total_loss += self.loss(y_batch[i], output[i])
            
            predictions = np.argmax(output, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            correct_predictions += np.sum(predictions == true_labels)
        
        # Set training mode back
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
        
        return total_loss / n_samples, correct_predictions / n_samples

def main():
    print("Loading CIFAR-10 data...")
    try:
        x_train, y_train, x_test, y_test = load_cifar10("assignment_2/data/cifar-10-batches-py/")
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
    except FileNotFoundError:
        print("CIFAR-10 data not found. Please download the data to 'assignment_2/data/cifar-10-batches-py/'")
        return
    
    # Preprocess data
    print("Preprocessing data...")
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
    
    # Create model
    print("Creating model...")
    model = create_cifar10_model()
    model.use(mse, mse_prime)
    model.set_optimizer(Adam(lr=0.001))
    
    # Split training data for validation
    val_split = 0.1
    n_val = int(len(x_train) * val_split)
    x_val = x_train[:n_val]
    y_val = y_train[:n_val]
    x_train = x_train[n_val:]
    y_train = y_train[n_val:]
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    
    # Train the model
    print("Starting training...")
    model.train(x_train, y_train, x_val, y_val, epochs=5, batch_size=32)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Final Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
