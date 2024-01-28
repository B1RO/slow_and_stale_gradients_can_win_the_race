import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

num_workers = 8
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load CIFAR-10 data
def grayscale_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 0.2989 * x[0] + 0.5870 * x[1] + 0.114 * x[2]),
        transforms.Lambda(lambda x: x.unsqueeze(0)),  # Add channel dimension
        transforms.Normalize((0.5,), (0.5,))
    ])


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=grayscale_transform())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=grayscale_transform())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Input channels: 3, Output channels: 6, Kernel size: 5
        self.pool = nn.MaxPool2d(2, 2)   # Max pooling with kernel size 2 and stride 2
        self.conv2 = nn.Conv2d(6, 16, 5) # Input channels: 6, Output channels: 16, Kernel size: 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Flattened size: 16*5*5, Output size: 120
        self.fc2 = nn.Linear(120, 84)    # Input size: 120, Output size: 84
        self.fc3 = nn.Linear(84, 10)      # Input size: 84, Output size: 3 (number of classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.12)

# Parameters for simulation
K=2
num_steps = 200000
shift = 0.0
scale = 0.02

def shifted_exponential(scale, shift):
    return np.random.exponential(scale) + shift

def compute_test_error(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    model.train()
    return total_loss / total_samples

# Simulate worker computation times with shifted exp distribution
remaining_times = [shifted_exponential(scale, shift) for _ in range(num_workers)]
staleness = [0 for _ in range(num_workers)]
stale_gradients = [None for _ in range(num_workers)]
train_loader_iter = iter(train_loader)
epochs = 0
time_budget = 360
time = 0
test_errors = []
times = []

# Training loop
for step in range(num_steps):
    end_of_epoch = False
    if time <= time_budget:
        if step % 300 == 0:  # Adjust the interval as needed
            test_error = compute_test_error(model, test_loader, criterion)
            test_errors.append(test_error)
            times.append(time)
            print(f"Step: {step}, Time: {time}, Test Error: {test_error}")
        # Identify the K workers with the least remaining time
        srt = np.argsort(remaining_times)
        fastest_workers = srt[:K]
        stale_workers = srt[K:]
        curr_iter_time = remaining_times[fastest_workers[K-1]] # time at which the K-th worker finishes
        time += curr_iter_time # Add to time counter

        # K fastest workers push their updates
        for worker in fastest_workers:
            remaining_times[worker] = 0
            optimizer.zero_grad()
            # Compute and apply fresh gradient
            if worker not in stale_workers or stale_gradients[worker] is None:
                try:
                    # Try to get the next batch
                    batch_x, batch_y = next(train_loader_iter)
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                except StopIteration:
                    end_of_epoch = True
                    epochs +=1
                    # If StopIteration is raised, restart the loader
                    train_loader_iter = iter(train_loader)
                    batch_x, batch_y = next(train_loader_iter)
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Perform the update
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)/K
                loss.backward()
            else:
                # Apply the stored stale gradient for this worker
                stale_gradient = stale_gradients[worker]
                # Load the stale gradients
                for name, param in model.named_parameters():
                    if param.grad is None:
                        param.grad = stale_gradient[name]
                    else:
                        param.grad += stale_gradient[name]

                stale_gradients[worker] = None

        # Accumulate all the gradients from the current iteration, then update the model parameters
        optimizer.step()

        for worker in stale_workers:
            # Get a batch from the worker's dataset
            try:
                # Try to get the next batch
                batch_x, batch_y = next(train_loader_iter)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            except StopIteration:
                if not end_of_epoch:
                  epochs += 1
                  end_of_epoch = True
                print(epochs)
                # If StopIteration is raised, restart the loader
                train_loader_iter = iter(train_loader)
                batch_x, batch_y = next(train_loader_iter)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Perform the update
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)/K
            loss.backward()
            # Store the stale gradients
            current_gradients = {name: param.grad.clone() for name, param in model.named_parameters()}
            stale_gradients[worker] = current_gradients



        # Other workers continue computing
        for i in range(num_workers):
            if remaining_times[i] > 0:
                remaining_times[i] -= curr_iter_time
            else: # Reset remaining time
                remaining_times[i] = shifted_exponential(scale, shift)

# Final evaluation of the model
model.eval()
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# Plotting the test error against training time
plt.plot(times, test_errors, label='Test Error')
plt.xlabel('Training Time (seconds)')
plt.ylabel('Test Error')
plt.title('Test Error vs Training Time')
plt.legend()
plt.show()