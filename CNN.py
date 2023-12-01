import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 100
batch_size = 2
learning_rate = 0.00001
learning_threshold = 0.1

transform = v2.Compose( 
    [v2.ToImage(),
     v2.CenterCrop(5000),
     v2.ToDtype(torch.uint8, scale=True),
     v2.Resize((512,512), antialias=True),
     v2.ToDtype(torch.float32, scale=True),
     v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

train_dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root='./data/test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('adequate', 'floded', 'no-data')

def imshow(imgs):
    imgs = imgs / 2 + 0.5 # unnormalize
    npimgs = imgs.numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()

# one batch of random training images
dataiter = iter(train_loader)
# print(dataiter.__next__())
images, labels = dataiter.__next__()
#imshow(torchvision.utils.make_grid(images[0:2], nrow=2))

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 512, 3)
        self.conv2 = nn.Conv2d(512, 256, 3)
        self.conv3 = nn.Conv2d(256, 128, 3)
        self.conv4 = nn.Conv2d(128, 64, 3)
        self.fc1 = nn.Linear(64*60*60, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # N, 3, 512, 512
        x = F.relu(self.conv1(x))   # -> N, 512, 510, 510
        x = self.pool(x)            # -> N, 512, 255, 255
        x = F.relu(self.conv2(x))   # -> N, 256, 253, 253
        x = self.pool(x)            # -> N, 256, 126, 126
        x = F.relu(self.conv3(x))   # -> N, 128, 124, 124
        x = self.pool(x)            # -> N, 128, 62, 62
        x = F.relu(self.conv4(x))   # -> N, 64, 60, 60
        x = torch.flatten(x, 1)     # -> N, 230_400
        x = F.relu(self.fc1(x))     # -> N, 64
        x = self.fc2(x)             # -> N, 3
        return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):

    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    print(f'[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}')

    if running_loss / n_total_steps <= learning_threshold:
        break

print('Finished Training')
PATH = './last-best-model.pth'

loaded_model = ConvNet()

try:
    loaded_model.load_state_dict(torch.load(PATH))
except:
    torch.save(model.state_dict(), PATH)
    loaded_model.load_state_dict(torch.load(PATH))

loaded_model.to(device)
loaded_model.eval()

with torch.no_grad():
    n_correct = 0
    n_correct2 = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        print(outputs)

        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()
        print(f'Actual Model -> label: {labels.tolist()}, predicted: {predicted.tolist()}')

        outputs2 = loaded_model(images)
        _, predicted2 = torch.max(outputs2, 1)
        n_correct2 += (predicted2 == labels).sum().item()
        print(f'Loaded Model -> label: {labels.tolist()}, predicted: {predicted2.tolist()}')

    acc_mem = 100.0 * n_correct / n_samples
    print(f'Accuracy of the model: {acc_mem} %')

    acc_loaded = 100.0 * n_correct2 / n_samples
    print(f'Accuracy of the loaded model: {acc_loaded} %')

    if (acc_mem > acc_loaded):
        torch.save(model.state_dict(), PATH)