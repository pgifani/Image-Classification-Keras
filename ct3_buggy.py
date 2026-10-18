"""
Bug hunt: 3-class CT organ classifier (bladder / heart / liver) with ImageFolder.
Needs the JPG folders from export_jpg.py:  data/ct3_jpg/{train,val,test}/{bladder,heart,liver}

Run:
    python ct3_buggy.py
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "ct3_jpg")
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---------------------------------------------------------------- transforms
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ---------------------------------------------------------------- data
train_ds = datasets.ImageFolder(os.path.join(DATA, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA, "val"), transform=train_tf)
print("classes:", train_ds.classes)
print("train:", len(train_ds), "val:", len(val_ds))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ---------------------------------------------------------------- model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.softmax(x, dim=1)


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------------------------------------------------------- training
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()
    for images, labels in train_loader:
        images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_ds)

    # ------------------------------------------------------------ validation
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            preds = outputs.argmax(dim=0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss /= len(val_loader)
    val_acc = correct / total

    print(f"epoch {epoch}: train loss {train_loss:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f}")
