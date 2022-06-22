import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import os
import numpy as np

# Implement CNN Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Architect Le-Net 5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = x.view(-1,  16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device('cuda')

PATH = 'saved_model/cnn.pth'
model = ConvNet()
model.load_state_dict(torch.load(PATH))
model.to(device)

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'dataset/fer'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'val']}

examples = iter(dataloaders['val'])

images, labels = examples.next()
images = images.to(device)
labels = labels.to(device)
print(images[0].shape, labels)


prediction = model(images[1])
result = prediction.argmax(dim=1)
if result == labels[0]:
    print("Marah")
elif result == labels[1]:
    print("Takut")
elif result == labels[2]:
    print("Senang")
elif result == labels[3]:
    print("Netral")
elif result == labels[4]:
    print("Sedih")
