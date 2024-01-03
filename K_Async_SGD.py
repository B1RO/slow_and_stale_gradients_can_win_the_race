import torch
import torch.nn as nn
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

# Linear model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.fc1(x.view(-1, 1024))
        return x

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)      

# Parameters for simulation
K0 = 1  # Initial value of K
K = K0
num_steps = 10000
shift = 1.0
scale = 0.02

def shifted_exponential(scale, shift):
    return np.random.exponential(scale) + shift

def compute_total_loss(model, data_loader, criterion):
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in data_loader:
        outputs = model(inputs.view(-1, 1024))
        loss = criterion(outputs, targets)
        total_loss += loss.item()  # Accumulate the loss
        total_samples += inputs.size(0)  # Count the total number of samples
    return total_loss / total_samples  # Return the average loss

# Initial loss
w0_loss = compute_total_loss(model, train_loader, criterion)
wstart_loss = w0_loss
wstart = model.state_dict()


time_counter = 0
t = 60
# Simulate worker computation times with shifted exp distribution
remaining_times = [shifted_exponential(scale, shift) for _ in range(num_workers)]
staleness = [0 for _ in range(num_workers)]
stale_gradients = [None for _ in range(num_workers)]
# Training loop
for step in range(num_steps):
    # Identify the K workers with the least remaining time
    srt = np.argsort(remaining_times)
    fastest_workers = srt[:K]
    stale_workers = srt[K:]
    curr_iter_time = remaining_times[fastest_workers[K-1]] # time at which the K-th worker finishes
    time_counter += curr_iter_time # Add to time counter
   
    # Update K if time_counter reaches threshold t
    if time_counter > t and K < num_workers:
        current_loss = compute_total_loss(model, train_loader, criterion)
        K = int(K0 * np.sqrt(w0_loss / current_loss))
        wstart = model.state_dict()  # Update wstart
        wstart_loss = current_loss
        K0 = K
        time_counter = 0 # Reset time counter
        
    
    # K fastest workers push their updates
    for worker in fastest_workers:
        remaining_times[worker] = 0
        optimizer.zero_grad()
        # Compute and apply fresh gradient
        if worker not in stale_workers or stale_gradients[worker] is None:
            batch_x, batch_y = next(iter(train_loader))

            # Perform the update
            outputs = model(batch_x.view(-1, 1024))
            loss = criterion(outputs, batch_y)/(K*batch_size)
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
            
            stored_gradients[worker] = None

    # Accumulate all the gradients from the current iteration, then update the model parameters
    optimizer.step()
        
    for worker in stale_workers:
        # Get a batch from the worker's dataset
        batch_x, batch_y = next(iter(train_loader))

        # Perform the update
        optimizer.zero_grad()
        outputs = model(batch_x.view(-1, 1024))
        loss = criterion(outputs, batch_y)/(K*batch_size)
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
# ...   