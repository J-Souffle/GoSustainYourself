import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import timm
from torch.cuda.amp import autocast, GradScaler
import time

# Mapping definitions (same as before)
CATEGORY_MAPPING = {
    "battery": "non-recyclable",       # Masuk ke kategori sampah non-daur ulang
    "biological": "non-recyclable",      # Sampah organik juga di sini
    "trash": "non-recyclable",
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
    "cardboard": "paper",
    "paper": "paper",
    "clothes": "fabric",               # Sesuai dengan kategori fabric-based
    "shoes": "fabric",                 # Masuk ke fabric juga
    "metal": "recyclable-inorganic",
    "plastic": "recyclable-inorganic"
}

GROUP_MAPPING = {
    "non-recyclable": 0,
    "glass": 1,
    "paper": 2,
    "fabric": 3,
    "recyclable-inorganic": 4
}

def get_class_index(object_name):
    key = object_name.strip().lower()
    for sub_str, group in CATEGORY_MAPPING.items():
        if sub_str in key:
            return GROUP_MAPPING[group]
    return None

class RecycleJPGDataset(Dataset):
    """
    Custom Dataset that loads image files (jpg, jpeg, png) and uses the parent folder name
    to determine the object's class.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for subdir, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(subdir, file)
                    object_name = os.path.basename(os.path.dirname(image_path))
                    label = get_class_index(object_name)
                    if label is not None:
                        self.samples.append((image_path, label))
                        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path)
        if image.mode == "P":
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

class XceptionClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(XceptionClassifier, self).__init__()
        # Load pretrained Xception as base model (head removed)
        # Note: timm's 'xception' model expects input size 299x299.
        self.base_model = timm.create_model('xception', pretrained=True, num_classes=0)
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Additional block similar to the Keras implementation:
        # A Conv2d layer with 512 filters, BatchNorm, ReLU, then MaxPool2d.
        self.additional_block = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2)
        )
        # Use adaptive pooling to flatten spatial dimensions.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # x should be of shape (batch, 3, 299, 299)
        x = self.base_model.forward_features(x)  # Get features with spatial dimensions
        x = self.additional_block(x)             # Expected shape: (batch, 512, H/2, W/2)
        x = self.avgpool(x)                      # (batch, 512, 1, 1)
        x = self.classifier(x)                   # (batch, num_classes)
        return x

def predict_image(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path)
    if image.mode == "P":
        image = image.convert("RGBA").convert("RGB")
    else:
        image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
    _, pred_class = torch.max(probs, 1)
    return pred_class.item(), probs[0, pred_class].item()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("amd") if hasattr(torch, 'amd') and torch.amd else device

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Update the data directory if needed
    data_dir = os.path.join(script_dir, 'data', 'garbage_classification')
    batch_size = 64
    num_epochs = 10

    # Update transforms for the Xception input size (299x299)
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataset from image files
    jpg_dataset = RecycleJPGDataset(data_dir, transform=train_transform)
    print(f"Found {len(jpg_dataset)} samples from image files.")
    total_size = len(jpg_dataset)
    # Use 20% for training and 20% for validation (ignore the rest)
    train_size = int(0.4 * total_size)
    val_size = int(0.03 * total_size)
    remaining_size = total_size - train_size - val_size
    train_dataset, val_dataset, _ = random_split(jpg_dataset, [train_size, val_size, remaining_size])
    print(f"Using {len(train_dataset)} samples for training and {len(val_dataset)} for validation.")

    # Overwrite validation transform
    if hasattr(val_dataset, 'dataset'):
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Instantiate the model with 5 classes
    model = XceptionClassifier(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    if device.type == 'cuda':
        scaler = GradScaler('cuda')

    for epoch in range(num_epochs):
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if device.type == 'cuda':
                with torch.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
                    
        epoch_loss = running_loss / train_size
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Time: {epoch_duration:.2f} sec")
        
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                if device.type == 'cuda':
                    with torch.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                        
        val_loss /= val_size
        accuracy = correct / val_size
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        scheduler.step(val_loss)
        
        # Early stopping if accuracy >= 97%
        if accuracy >= 0.94:
            print("Reached 94% accuracy. Stopping training.")
            break

    # Save model after training ends.
    model_save_path = os.path.join(script_dir, 'xception_garbage_classifier.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Example inference
    test_image_path = os.path.join(script_dir, 'data', 'Main Dataset', 'glass', 'example.png')
    pred, prob = predict_image(model, test_image_path, val_transform, device)
    inv_map = {v: k for k, v in GROUP_MAPPING.items()}
    result = inv_map.get(pred, "Unknown")
    print(f"Image: {test_image_path}")
    print(f"Prediction: {result} (confidence: {prob:.4f})")