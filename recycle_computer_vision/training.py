import os
import xml.etree.ElementTree as ET
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset

from torchvision import transforms

# Define the set of recyclable object names (in lowercase)
RECYCLABLE_OBJECTS = {"bottle", "can", "paper", "cardboard", "plastic", "jar", "glass"}

def diagnose_recyclability(object_name):
    """
    Given an object name string, return 1 if recyclable, 0 if nonrecyclable.
    """
    return 1 if object_name.strip().lower() in RECYCLABLE_OBJECTS else 0

class RecycleJPGDataset(Dataset):
    """
    Custom Dataset that loads image files (jpg, jpeg, png) and uses the parent
    folder name as the object's name to determine its recyclability label.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of tuples: (image_path, label)
        for subdir, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(subdir, file)
                    # Assume the parent folder's name is indicative of the object name
                    object_name = os.path.basename(os.path.dirname(image_path))
                    label = diagnose_recyclability(object_name)
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
        return image, torch.tensor(label, dtype=torch.float)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            # Block 4 (additional complexity)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def predict_image(model, image_path, transform, device):
    """
    Given a model, an image path, transform, and device,
    loads the image, applies transforms, and returns the model's prediction.
    """
    model.eval()
    image = Image.open(image_path)
    if image.mode == "P":
        image = image.convert("RGBA").convert("RGB")
    else:
        image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output)
    pred = 1 if prob.item() >= 0.5 else 0
    return pred, prob.item()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use the script's directory as the base
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir  # update this if your image folder is elsewhere
    img_height, img_width = 224, 224
    batch_size = 32
    num_epochs = 5

    train_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataset solely from image files (JPG, JPEG, PNG)
    jpg_dataset = RecycleJPGDataset(data_dir, transform=train_transform)
    
    print(f"Found {len(jpg_dataset)} samples from image files.")
    total_size = len(jpg_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(jpg_dataset, [train_size, val_size])
    # Update validation transform for the split dataset if possible
    if hasattr(val_dataset, 'dataset'):
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / train_size
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}")
        
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.unsqueeze(1).float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) >= 0.5
                correct += (preds.float() == labels).sum().item()
                
        val_loss /= val_size
        accuracy = correct / val_size
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        scheduler.step(val_loss)

    torch.save(model.state_dict(), os.path.join(script_dir, 'recyclable_classifier.pth'))
    
    # Example inference remains unchanged
    test_image_path = os.path.join(script_dir, 'hello', 'data', 'Main Dataset', 'BOTTLE', 'BOTTLE (2).png')
    pred, prob = predict_image(model, test_image_path, val_transform, device)
    result = "Recyclable" if pred == 1 else "Non-Recyclable"
    print(f"Image: {test_image_path}")
    print(f"Prediction: {result} (confidence: {prob:.4f})")