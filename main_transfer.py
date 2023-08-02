import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm import tqdm
from tqdm import trange

import pandas as pd
import math

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EPOCH = 20
BS = 64
IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet18_tf = models.resnet18(pretrained=True).to(device)
resnet18_tf.fc = nn.Sequential(nn.Linear(resnet18_tf.fc.in_features, 1), nn.Sigmoid())
resnet18_tf.__name__ = "ResNet18"

resnet50_tf = models.resnet50(pretrained=True).to(device)
resnet50_tf.fc = nn.Sequential(nn.Linear(resnet50_tf.fc.in_features, 1), nn.Sigmoid())
resnet50_tf.__name__ = "ResNet50"

densenet_tf = models.densenet161(pretrained=True).to(device)
densenet_tf.classifier = nn.Sequential(nn.Linear(densenet_tf.classifier.in_features, 1), nn.Sigmoid())
densenet_tf.__name__ = "Dense"

shufflenet_tf = models.shufflenet_v2_x1_0(pretrained=True).to(device)
shufflenet_tf.fc = nn.Sequential(nn.Linear(shufflenet_tf.fc.in_features, 1), nn.Sigmoid())
shufflenet_tf.__name__ = "Shufflenet"

mobilenet_tf = models.mobilenet_v2(pretrained=True).to(device)
mobilenet_tf.classifier[1] = nn.Sequential(nn.Linear(mobilenet_tf.classifier[1].in_features, 1), nn.Sigmoid())
mobilenet_tf.__name__ = "Mobilenet"

resnext50_32x4d_tf = models.resnext50_32x4d(pretrained=True).to(device)
resnext50_32x4d_tf.fc = nn.Sequential(nn.Linear(resnext50_32x4d_tf.fc.in_features, 1), nn.Sigmoid())
resnext50_32x4d_tf.__name__ = "Resnext50_32x4d"

wide_resnet50_2_tf = models.wide_resnet50_2(pretrained=True).to(device)
wide_resnet50_2_tf.fc = nn.Sequential(nn.Linear(wide_resnet50_2_tf.fc.in_features, 1), nn.Sigmoid())
wide_resnet50_2_tf.__name__ = "Wide_resnet50_2"

mnasnet_tf = models.mnasnet1_0(pretrained=True).to(device)
mnasnet_tf.classifier[1] = nn.Sequential(nn.Linear(mnasnet_tf.classifier[1].in_features, 1), nn.Sigmoid())
mnasnet_tf.__name__ = "mnasnet"

# Create a list of all the models
models_tf = [resnet18_tf, resnet50_tf, densenet_tf, shufflenet_tf, mobilenet_tf, resnext50_32x4d_tf, wide_resnet50_2_tf]


resnet18 = models.resnet18().to(device)
resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features, 1), nn.Sigmoid())
resnet18.__name__ = "ResNet18"

resnet50 = models.resnet50().to(device)
resnet50.fc = nn.Sequential(nn.Linear(resnet50.fc.in_features, 1), nn.Sigmoid())
resnet50.__name__ = "ResNet50"

densenet = models.densenet161().to(device)
densenet.classifier = nn.Sequential(nn.Linear(densenet.classifier.in_features, 1), nn.Sigmoid())
densenet.__name__ = "Dense"

shufflenet = models.shufflenet_v2_x1_0().to(device)
shufflenet.fc = nn.Sequential(nn.Linear(shufflenet.fc.in_features, 1), nn.Sigmoid())
shufflenet.__name__ = "Shufflenet"

mobilenet = models.mobilenet_v2().to(device)
mobilenet.classifier[1] = nn.Sequential(nn.Linear(mobilenet.classifier[1].in_features, 1), nn.Sigmoid())
mobilenet.__name__ = "Mobilenet"

resnext50_32x4d = models.resnext50_32x4d().to(device)
resnext50_32x4d.fc = nn.Sequential(nn.Linear(resnext50_32x4d.fc.in_features, 1), nn.Sigmoid())
resnext50_32x4d.__name__ = "Resnext50_32x4d"

wide_resnet50_2 = models.wide_resnet50_2().to(device)
wide_resnet50_2.fc = nn.Sequential(nn.Linear(wide_resnet50_2.fc.in_features, 1), nn.Sigmoid())
wide_resnet50_2.__name__ = "Wide_resnet50_2"

mnasnet = models.mnasnet1_0().to(device)
mnasnet.classifier[1] = nn.Sequential(nn.Linear(mnasnet.classifier[1].in_features, 1), nn.Sigmoid())
mnasnet.__name__ = "mnasnet"

# Create a list of all the models
models = [resnet18, resnet50, densenet, shufflenet, mobilenet, resnext50_32x4d, wide_resnet50_2]

# Define transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.3), ratio=(0.7, 1.3)),  # Random shifts and zooms
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),    # Adjust brightness, contrast, saturation, and hue
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=40),  # Rotate the image by a random angle up to 15 degrees
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std for a single channel
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize images to 128x128
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std for a single channel
])

transform_other = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.3), ratio=(0.7, 1.3)),  # Random shifts and zooms
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),    # Adjust brightness, contrast, saturation, and hue
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=40),  # Rotate the image by a random angle up to 15 degrees
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std for a single channel
])

transform_other_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize images to 128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std for a single channel
])

import os
if not os.path.exists('models'):
    os.makedirs('models')


# Load data

dataset_other = torchvision.datasets.ImageFolder(root='Trainset', transform=transform_other)
dataset_other_test = torchvision.datasets.ImageFolder(root='Testset', transform=transform_other_test)

# Create a balanced sampler
class_count = [0, 0]  # assuming binary classification
for _, index in dataset:
    class_count[index] += 1
weights = 1. / torch.tensor(class_count, dtype=torch.float)
sample_weights = weights[dataset.targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
print(class_count)

dataloader_other = DataLoader(dataset_other, batch_size=BS, sampler=sampler)
dataloader_other_test = DataLoader(dataset_other_test, batch_size=BS)


from torch import optim

# Define a loss function and optimizer
criterion = nn.BCELoss()


def train_model(model, dataloader, dataloader_test, criterion, optimizer, scheduler, device, num_epochs):
    model.train()
    results = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0
        t = trange(len(dataloader), desc=f'Epoch {epoch+1}', leave=True)
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.type_as(outputs)
            preds = torch.where(outputs > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            loss = criterion(outputs, labels.unsqueeze(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.view((-1)) == labels.type(torch.int64).data)
            total += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
            t.set_description(f'Epoch {epoch+1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            t.refresh()

        test_loss, test_acc, precision, recall, f1 = test_model(model, dataloader_test, criterion, device)
        model.train()
        print(test_loss, test_acc)

        results.append((epoch, epoch_loss, epoch_acc.item(), test_loss, test_acc, precision, recall, f1))
        #results.append((epoch, epoch_loss, epoch_acc.item()))
        scheduler.step()

    return results




def test_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            labels = labels.type_as(outputs)
            preds = torch.where(outputs > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss = criterion(outputs, labels.unsqueeze(-1))
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.view((-1)) == labels.type(torch.int64).data)
            total += inputs.size(0)

        loss = running_loss / total
        acc = running_corrects.double() / total

        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

    return loss, acc.item(), precision, recall, f1


comparison_results = []

for model in models_tf:
    model_name = model.__name__
    print(f"Training model: {model_name}")

    if not os.path.exists(f'models_tf/{model_name}'):
        os.makedirs(f'models_tf/{model_name}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1.0)

    model.to(device)

    train_results = train_model(model, dataloader_other, dataloader_other_test, criterion, optimizer, scheduler, device, num_epochs=EPOCH)
    train_df = pd.DataFrame(train_results, columns=['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Precision', 'Recall', 'F1'])
    train_df.to_csv(f'models_tf/{model_name}/train_results.csv', index=False)

    test_loss, test_acc, precision, recall, f1 = test_model(model, dataloader_other_test, criterion, device)
    test_df = pd.DataFrame([(test_loss, test_acc, precision, recall, f1)], columns=['Test Loss', 'Test Accuracy', 'Precision', 'Recall', 'F1'])
    test_df.to_csv(f'models_tf/{model_name}/test_results.csv', index=False)

    torch.save(model.state_dict(), f'models_tf/{model_name}/model.pt')

    comparison_results.append((model_name, train_results[-1][1], train_results[-1][2], test_loss, test_acc))

comparison_df = pd.DataFrame(comparison_results,
                             columns=['Model', 'Last Train Loss', 'Last Train Accuracy', 'Test Loss', 'Test Accuracy'])
comparison_df.to_csv('comparison_results_tf.csv', index=False)

print("Finished Training")

for model in models:
    model_name = model.__name__
    print(f"Training model: {model_name}")

    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1.0)

    model.to(device)

    train_results = train_model(model, dataloader_other, dataloader_other_test, criterion, optimizer, scheduler, device, num_epochs=EPOCH)
    train_df = pd.DataFrame(train_results, columns=['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Precision', 'Recall', 'F1'])
    train_df.to_csv(f'models/{model_name}/train_results.csv', index=False)

    test_loss, test_acc, precision, recall, f1 = test_model(model, dataloader_other_test, criterion, device)
    test_df = pd.DataFrame([(test_loss, test_acc, precision, recall, f1)], columns=['Test Loss', 'Test Accuracy', 'Precision', 'Recall', 'F1'])
    test_df.to_csv(f'models/{model_name}/test_results.csv', index=False)

    torch.save(model.state_dict(), f'models/{model_name}/model.pt')

    comparison_results.append((model_name, train_results[-1][1], train_results[-1][2], test_loss, test_acc))

comparison_df = pd.DataFrame(comparison_results,
                             columns=['Model', 'Last Train Loss', 'Last Train Accuracy', 'Test Loss', 'Test Accuracy'])
comparison_df.to_csv('comparison_results.csv', index=False)

print("Finished Training")


