import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from cv2.cv2 import imshow
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Mengatasi error konflik library
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Device config
device = torch.device('cuda')

# Load models
PATH = 'saved_model/cnn_pretrained.pth'
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
model.load_state_dict(torch.load(PATH))
model.to(device)

# Load data
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
    'test': transforms.Compose([
        transforms.Resize(48),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'dataset/fer'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes


# Create function to plot image
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Create model prediction visualization function
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')

                if (preds[j] == 0):
                    label_name = "Stres"
                elif (preds[j] == 1):
                    label_name = "Lumayan Stress"
                elif (preds[j] == 2):
                    label_name = "Senang"
                elif (preds[j] == 3):
                    label_name = "Tidak Stress"
                elif (preds[j] == 4):
                    label_name = "Stress"
                ax.set_title(f'predicted: {label_name}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

visualize_model(model)
plt.ioff()
plt.show()

# Single prediction
image_path = "dataset/fer/test/angry/DMUbjq2UjJcG3umGv3Qjjd.jpeg"
def transform_image(path):
    transform = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    image = Image.open(path)
    image = transform(image).float()
    image = image.unsqueeze(0)
    return image.cuda()

# predict
def get_prediction(image_tensor):
    outputs = model(image_tensor)
        # max returns (value ,index)
    _, predicted = torch.max(outputs, 1)
    return predicted
# image = transform_image(image_path)
# prediction = get_prediction(image)
# print(prediction)

