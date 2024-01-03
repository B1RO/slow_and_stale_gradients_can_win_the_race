# Importieren der PyTorch-Bibliotheken.
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


# Transformations-Pipeline für die Bilder des CIFAR-100 Datensatzes.
# 'transforms.Compose' kombiniert mehrere Transformationsschritte.
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #
])
# Laden des CIFAR-100 Trainingssatzes. 'train=True' wählt den Trainingssatz aus.
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# Laden des CIFAR-100 Testsatzes. 'train=False' wählt den Testsatz aus.
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Liste der Label-Indizes für die Superklasse 'aquatic mammals' im CIFAR-100 Datensatz.
aquatic_mammals_labels = [4, 30, 55, 72, 95] 

# filter data and save data in file 

def filter_aquatic_mammals_test(dataset, labels):
    class_indices = [i for i, label in enumerate(dataset.targets) if label in labels]
    dataset.data = dataset.data[class_indices]
    dataset.targets = [dataset.targets[i] for i in class_indices]
    with open('test_aquatic_data.pkl', 'wb') as file:
        pickle.dump(dataset, file)

def filter_aquatic_mammals_train(dataset, labels):
    class_indices = [i for i, label in enumerate(dataset.targets) if label in labels]
    dataset.data = dataset.data[class_indices]
    dataset.targets = [dataset.targets[i] for i in class_indices]
    with open('train_aquatic_data.pkl', 'wb') as file:
        pickle.dump(dataset, file)

# Anwendung der Filterfunktion auf den Trainings- und Testsatz.
filter_aquatic_mammals_train(trainset, aquatic_mammals_labels)
filter_aquatic_mammals_test(testset, aquatic_mammals_labels)


# get saved data

with open('test_aquatic_data.pkl', 'rb') as file:
    testset = pickle.load(file)

with open('train_aquatic_data.pkl', 'rb') as file:
    trainset = pickle.load(file)



# Erstellen eines DataLoader-Objekts für den Testsatz.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)