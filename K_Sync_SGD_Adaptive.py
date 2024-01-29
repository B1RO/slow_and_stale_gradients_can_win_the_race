import math
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

def shifted_exponential(scale, shift):
    return np.random.exponential(scale) + shift

def compute_total_loss(model, data_loader, criterion):
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in data_loader:

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        total_loss += loss.item()* inputs.size(0)  # Accumulate the loss
        total_samples += inputs.size(0)

    return total_loss / total_samples 

def solve_quadratic(a, b, c):
    # Calculate the discriminant
    d = b**2 - 4*a*c

    # Check if discriminant is negative
    if d < 0:
        return "No real roots"

    # Calculate two solutions
    sol1 = (-b + math.sqrt(d)) / (2*a)
    sol2 = (-b - math.sqrt(d)) / (2*a)

    # Check if solutions are positive integers
    roots = []
    if sol1 > 0:
        roots.append(sol1)
    if sol2 > 0  and sol2 != sol1:
        roots.append(sol2)

    return roots if roots else "No positive roots"


def K_sync_SGD(K, num_steps=200000, t=60, time_budget=100, lr=0.01, scale=0.02, shift=0.0, evaluation_interval=300, Adaptive=False):
    # Initialize model, criterion, optimizer
    K0 = K
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    w0_loss = compute_total_loss(model, train_loader, criterion)
    wstart_loss = w0_loss
    wstart = model.state_dict()

    # Initialize variables
    train_loader_iter = iter(train_loader)
    epochs = 0
    time = 0
    time_counter = 0
    test_errors = []
    train_errors = []
    times = []

    # Training loop
    for step in range(num_steps):
        if time < time_budget:
            if step % evaluation_interval == 0:
                test_error = compute_total_loss(model, test_loader, criterion)
                test_errors.append(test_error)
                times.append(time)
                print(f"Step: {step}, Time: {time}, Test Error: {test_error}")
                train_error = compute_total_loss(model, train_loader, criterion)
                train_errors.append(train_error)
                print(f"Step: {step}, Time: {time}, Train Error: {train_error}")
                model.train()

            # Identify the K workers with the least remaining time
            remaining_times = [shifted_exponential(scale, shift) for _ in range(num_workers)]
            fastest_workers = np.argsort(remaining_times)[:K]
            curr_iter_time = remaining_times[fastest_workers[K-1]]
            time_counter += curr_iter_time
            time += curr_iter_time

            # Update K if Adaptive is True and conditions are met
            if Adaptive and time_counter > t and K < num_workers:
                current_loss = compute_total_loss(model, train_loader, criterion)
                a = 1 
                b = K0**2 * w0_loss / ((num_workers - K0) * current_loss)
                c = -K0**2 * w0_loss * num_workers / ((num_workers - K0) * current_loss)
                roots = solve_quadratic(a, b, c)
                if isinstance(roots, list) and roots:
                    K = round(min(roots))
                    K = min(K, num_workers)  # Ensure K does not exceed num_workers
                wstart = model.state_dict()
                wstart_loss = current_loss
                time_counter = 0

            # K fastest workers push their updates
            for worker in fastest_workers:
                remaining_times[worker] = 0
                optimizer.zero_grad()
                try:
                    batch_x, batch_y = next(train_loader_iter)
                except StopIteration:
                    epochs += 1
                    train_loader_iter = iter(train_loader)
                    batch_x, batch_y = next(train_loader_iter)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y) / K
                loss.backward()
                optimizer.step()

    # Final evaluation of the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct // total
    print(f'Accuracy of the network on the 10000 test images: {accuracy} %')

    return model, test_errors, train_errors, times

# Compare the performances
model_2, test_errors_2, train_errors_2, times_2 = K_sync_SGD(K=2)
model_4, test_errors_4, train_errors_4, times_4 = K_sync_SGD(K=4)
model_8, test_errors_8, train_errors_8, times_8 = K_sync_SGD(K=8)
model_ada, test_errors_ada, train_errors_ada, times_ada = K_sync_SGD(K=1, Adaptive=True)

plt.plot(times_8, test_errors_8, label='K=8')
plt.plot(times_4, test_errors_4, label='K=4')
plt.plot(times_2, test_errors_2, label='K=2')
plt.plot(times_ada, test_errors_ada, label='AdaSync')
plt.xlabel('Training Time (seconds)')
plt.ylabel('Test Error')
plt.title('Test Error vs Training Time')
plt.legend()
plt.show()



