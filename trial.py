import os
import copy
import time
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================================================
# CONFIG
# =========================================================
 
IMAGE_ROOT = "/global/homes/m/mryu2/DeepImageClassification/data/Refuge/REFUGE_images"

TRAIN_CSV = "./data/Refuge/train_annotations.csv"
VAL_CSV   = "./data/Refuge/val_annotations.csv"
TEST_CSV  = "./data/Refuge/test_annotations.csv"

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

# Save path
SAVE_PATH = "best_resnet50_refuge.pth"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# =========================================================
# TRANSFORM
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMAGE_SIZE),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================================================
# DATASET
# =========================================================

class RefugeDataset(Dataset):

    def __init__(self, csv_file, image_root, transform=None):

        self.df = pd.read_csv(csv_file)

        self.image_root = image_root

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        label = int(row["label"])

        file_name = row["fileName"]

        image_path = os.path.join(
            self.image_root,
            file_name
        )

        # 디버깅용
        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# DATALOADER
# =========================================================

train_dataset = RefugeDataset(
    csv_file=TRAIN_CSV,
    image_root=IMAGE_ROOT,
    transform=train_transform
)

val_dataset = RefugeDataset(
    csv_file=VAL_CSV,
    image_root=IMAGE_ROOT,
    transform=val_transform
)

test_dataset = RefugeDataset(
    csv_file=TEST_CSV,
    image_root=IMAGE_ROOT,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# =========================================================
# MODEL
# =========================================================

model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V2
)

in_features = model.fc.in_features

model.fc = nn.Linear(
    in_features,
    2
)

model = model.to(DEVICE)


# =========================================================
# LOSS
# =========================================================

# REFUGE는 imbalance 심하므로 weight 추천
class_weights = torch.tensor(
    [1.0, 9.0]
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)


# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in tqdm(loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for images, labels in tqdm(loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            probs = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = running_loss / total

    epoch_acc = correct / total

    auc = roc_auc_score(
        all_labels,
        all_probs
    )

    return epoch_loss, epoch_acc, auc


# =========================================================
# TRAIN LOOP
# =========================================================

best_auc = 0.0

best_model_wts = copy.deepcopy(
    model.state_dict()
)


for epoch in range(NUM_EPOCHS):
    stime = time.time()
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

    # -----------------------------
    # Train
    # -----------------------------

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader
    )

    # -----------------------------
    # Validation
    # -----------------------------

    val_loss, val_acc, val_auc = evaluate(
        model,
        val_loader
    )

    scheduler.step()

    # -----------------------------
    # Print
    # -----------------------------

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    print(" time = ", time.time() - stime)
    # -----------------------------
    # Save best model
    # -----------------------------

    if val_auc > best_auc:

        best_auc = val_auc

        best_model_wts = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_model_wts,
            SAVE_PATH
        )

        print(
            f"Best model saved "
            f"(AUC={best_auc:.4f})"
        )


# =========================================================
# LOAD BEST MODEL
# =========================================================

print("\nLoading Best Model...")

model.load_state_dict(
    torch.load(SAVE_PATH)
)


# =========================================================
# TEST
# =========================================================

test_loss, test_acc, test_auc = evaluate(
    model,
    test_loader
)

print("\n===== TEST RESULT =====")

print(f"Test Loss : {test_loss:.4f}")
print(f"Test Acc  : {test_acc:.4f}")
print(f"Test AUC  : {test_auc:.4f}")