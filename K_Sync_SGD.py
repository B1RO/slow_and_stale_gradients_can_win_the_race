import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

num_workers = 8
batch_size = 32


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


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.12)

# Parameters for simulation
K0 = 1  # Initial value of K
K = K0
num_steps = 100000
shift = 0.0
scale = 0.02

def shifted_exponential(scale, shift):
    return np.random.exponential(scale) + shift

def compute_total_loss(model, data_loader, criterion):
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in data_loader:
        
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
# Initial loss
w0_loss = compute_total_loss(model, train_loader, criterion)
wstart_loss = w0_loss
wstart = model.state_dict()

epochs = 0
time_counter = 0
t = 60
# Create an iterator from train_loader
train_loader_iter = iter(train_loader)
# Training loop
for step in range(num_steps):
    if epochs <= 250:
        # Identify the K workers with the least remaining time
        # Simulate worker computation times with shifted exp distribution
        remaining_times = [shifted_exponential(scale, shift) for _ in range(num_workers)]
        fastest_workers = np.argsort(remaining_times)[:K]
        curr_iter_time = remaining_times[fastest_workers[K-1]] # time at which the K-th worker finishes
        time_counter += curr_iter_time # Add to time counter

        # Update K if time_counter reaches threshold t
        if time_counter > t and K < num_workers:
            current_loss = compute_total_loss(model, train_loader, criterion)
            print(f'Current Loss: {current_loss}')
            a = 1 
            b = K0**2 * w0_loss/((num_workers - K0)*current_loss)
            c = -K0**2 * w0_loss * num_workers /((num_workers - K0)*current_loss)
            roots = solve_quadratic(a, b, c)
            print(f'a: {a}, b: {b}, c: {c}')
            print(f'Roots: {roots}')
            if isinstance(roots, list) and roots:  # Check if roots are positive 
                
                K = round(min(roots))  # Update K to the smallest positive integer root
                if K>=8:
                    K=8
            wstart = model.state_dict()  # Update wstart
            wstart_loss = current_loss
            K0 = K
            time_counter = 0 # Reset time counter
            print(f'K: {K}')
            
        optimizer.zero_grad()
        # K fastest workers push their updates
        for worker in fastest_workers:
            remaining_times[worker] = 0
            try:
                # Try to get the next batch
                batch_x, batch_y = next(train_loader_iter)
            except StopIteration:
                epochs +=1
                print(epochs)
                # If StopIteration is raised, restart the loader
                train_loader_iter = iter(train_loader)
                batch_x, batch_y = next(train_loader_iter)
            # Perform the update
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)/K
            loss.backward()
        optimizer.step()    
    


# Final evaluation of the model
model.eval()  
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')