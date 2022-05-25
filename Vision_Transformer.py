import numpy as np
import torch
import torchvision
import torch.nn as nn
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce


class ImageEmbedding(nn.Module):
    def __init__(self, channels, resolution, patch_resolution, emb_dim):
        super().__init__()
        self.rearrange = Rearrange('batch channel (width patch_width) (height patch_height) -> '
                                   'batch (width height) (patch_width patch_height channel)',
                                   patch_width=patch_resolution, patch_height=patch_resolution)
        self.fc = nn.Linear(channels * patch_resolution * patch_resolution, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.randn((resolution // patch_resolution)**2 + 1, emb_dim))

    def forward(self, x):
        batch, channel, width, height = x.shape
        x = self.rearrange(x)
        x = self.fc(x)
        c = repeat(self.cls_token, '() 1 emb_dim -> batch 1 emb_dim', batch=batch)
        x = torch.cat((c, x), dim=1)
        x = x + self.pos_emb
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.root_dk = (emb_dim // num_heads) ** -0.5
        self.value = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = rearrange(query, 'batch query (num_heads x) -> batch num_heads query x', num_heads=self.num_heads)
        key = rearrange(key, 'batch key (num_heads x) -> batch num_heads x key', num_heads=self.num_heads)
        value = rearrange(value, 'batch value (num_heads x) -> batch num_heads value x', num_heads=self.num_heads)

        weight = torch.matmul(query, key)
        weight = weight * self.root_dk
        attention = torch.softmax(weight, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, value)
        context = rearrange(context, 'batch num_heads query x -> batch query (num_heads x)')

        output = self.linear(context)
        return output, attention


class MultiLayerPerceptron(nn.Module):
    def __init__(self, emb_dim, expansion, dropout):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim, expansion * emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(expansion * emb_dim, emb_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_heads, expansion, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.mha = MultiHeadedAttention(emb_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MultiLayerPerceptron(emb_dim, expansion, dropout)
        self.residual = nn.Dropout(dropout)

    def forward(self, x):
        tmp = self.norm1(x)
        tmp, attention = self.mha(tmp)
        x = tmp + self.residual(x)

        tmp = self.norm2(x)
        tmp = self.mlp(tmp)
        x = tmp + self.residual(x)
        return x, attention


class VisionTransformer(nn.Module):
    def __init__(self, channels, resolution, patch_resolution, emb_dim,
                 num_heads, expansion, dropout, num_layers, num_class):
        super().__init__()
        self.img_emb = ImageEmbedding(channels, resolution, patch_resolution, emb_dim)
        self.encoder = nn.ModuleList([TransformerEncoder(emb_dim, num_heads, expansion, dropout)
                                      for _ in range(num_layers)])
        self.reduce = Reduce('b n e -> b e', reduction='mean')
        self.norm = nn.LayerNorm(emb_dim)
        self.classification = nn.Linear(emb_dim, num_class)

    def forward(self, x):
        x = self.img_emb(x)
        attentions = []
        for encoder in self.encoder:
            x, att = encoder(x)
            attentions.append(att)

        x = self.reduce(x)
        x = self.norm(x)
        x = self.classification(x)
        return x, attentions


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VisionTransformer(1, 28, 4, 16, 2, 4, 0.2, 3, 10)
model.to(device)
num_epochs = 50
batch_size = 64
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5), (0.5, 0.5)),
])
dataset_train = torchvision.datasets.MNIST('MNIST', train=True, download=True, transform=transforms)
dataset_test = torchvision.datasets.MNIST('MNIST', train=False, download=True, transform=transforms)
dataloader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(range(0, len(dataset_train) * 4//5))
)
dataloader_valid = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(range(len(dataset_train) * 4//5, len(dataset_train)))
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size
)

early_stopping = EarlyStopping(patience=5, delta=0)
for epoch in range(num_epochs):
    print("Epoch", epoch + 1)
    model.train()

    train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
    for i, (inputs, labels) in enumerate(dataloader_train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.shape[0]
            train_acc += torch.sum(preds == labels.data)
    train_acc /= dataset_train.data.shape[0] * 0.8
    print("train loss: {:.4f} train acc: {:.4f}".format(train_loss, train_acc))

    model.eval()
    for i, (inputs, labels) in enumerate(dataloader_valid):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.shape[0]
            val_acc += torch.sum(preds == labels.data)
    val_acc /= dataset_train.data.shape[0] * 0.2
    print("val loss: {:.4f} val acc: {:.4f}".format(val_loss, val_acc))
    early_stopping(val_loss)
    if early_stopping.time_to_stop:
        print("Early Stopping")
        break

model.eval()
test_acc = 0.0
for i, (inputs, labels) in enumerate(dataloader_test):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.set_grad_enabled(False):
        outputs, _ = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_acc += torch.sum(preds == labels.data)
test_acc /= dataset_test.data.shape[0]
print("test acc: {:.4f}".format(test_acc))
