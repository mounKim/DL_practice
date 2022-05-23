import csv
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as t
import numpy as np
from statistics import mean
from torchsummary import summary
from tqdm import tqdm


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class Block(nn.Module):

    def __init__(self, in_planes, out_planes, downsample=None):
        super(Block, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)   # inplace=True이므로 input 자체를 수정해서 메모리에 효율적
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3, self.in_planes)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self.make_layer(16, 10)
        self.layer2 = self.make_layer(32, 10)
        self.layer3 = self.make_layer(64, 10)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, out_planes, blocks):
        downsample = nn.Sequential(
            conv1x1(self.in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
        )
        layers = [Block(self.in_planes, out_planes, downsample)]
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(Block(self.in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.time_to_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.time_to_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def train():
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []

    early_stopping = EarlyStopping(patience=5, delta=0)
    test_acc_list = []

    for epoch in range(num_epochs):
        print('Epoch', epoch + 1)
        model.train()
        correct_train = 0
        correct_test = 0
        batch_losses = []

        for batch, targets in tqdm(train_loader):
            batch = batch.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == targets.data)

        train_losses.append(mean(batch_losses))
        scheduler.step()

        model.eval()
        y_pred = []
        with torch.no_grad():
            for batch, targets in tqdm(test_loader):
                batch = batch.to(device)
                targets = targets.to(device)
                outputs = model(batch)
                loss = criterion(outputs, targets)
                test_losses.append(loss.item())
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
                _, preds = torch.max(outputs, 1)
                correct_test += torch.sum(preds == targets.data)

        train_acc = correct_train.item() / train_set.data.shape[0]
        test_acc = correct_test.item() / test_set.data.shape[0]

        print('Training loss: {:.4f}'.format(np.average(train_losses)))
        print('Test loss: {:.4f}'.format(np.average(test_losses)))
        print('Training accuracy: {:.2f}%'.format(float(train_acc) * 100))
        print('Test accuracy: {:.2f}%\n'.format(float(test_acc) * 100))

        test_acc_list.append(float(test_acc) * 100)
        early_stopping(np.average(test_losses))
        if early_stopping.time_to_stop:
            print("Early Stopping")
            break

    return test_acc_list, y_pred


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
learning_rate = 0.1
num_epochs = 1000

'''
temp = Resnet()
summary(temp, (3, 32, 32))
'''

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transforms_train = t.Compose([
    t.RandomCrop(32, padding=4),
    t.RandomHorizontalFlip(),
    t.ToTensor(),
    t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transforms_test = t.Compose([
    t.ToTensor(),
    t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_set = torchvision.datasets.CIFAR10('./CIFAR-10', train=True, download=True, transform=transforms_train)
test_set = torchvision.datasets.CIFAR10('./CIFAR-10', train=False, download=True, transform=transforms_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = Resnet()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2400, gamma=0.1)

test_acc_list, y_pred = train()

with open("./submit.csv", mode="w") as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerow(["ID", "Category"])
    for i, label in enumerate(y_pred):
        writer.writerow([i + 1, classes[label]])
