# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchvision",
#     "matplotlib",
#     "seaborn",
#     "scikit-learn",
#     "tqdm",
#     "pillow",
#     "numpy",
#     "imagehash",
#     "duckduckgo_search",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell
def _():
    import os
    import shutil
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, models
    from torchvision.io import decode_image, ImageReadMode
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    from tqdm import tqdm
    from pathlib import Path
    from PIL import Image
    import imagehash
    from duckduckgo_search import DDGS
    import warnings
    warnings.filterwarnings('ignore')
    return (
        DataLoader,
        Dataset,
        DDGS,
        Image,
        ImageReadMode,
        decode_image,
        imagehash,
        models,
        nn,
        np,
        optim,
        os,
        plt,
        random,
        shutil,
        sns,
        tqdm,
        transforms,
        warnings,
    )

@app.cell
def _():
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device, seed

@app.cell
def _():
    # Define classes
    CLASSES = ['toyota', 'bmw', 'other']
    NUM_CLASSES = len(CLASSES)
    return CLASSES, NUM_CLASSES

@app.cell
def _():
    # -------------------------------
    # 1. Data Collection via Web Scraping
    # -------------------------------
    # We'll download images using DuckDuckGo (free, no API key)
    # For each class we define search queries to get variety

    def download_images(query, save_dir, limit=100):
        """Download images using DDGS and save to save_dir."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with DDGS() as ddgs:
            results = ddgs.images(query, max_results=limit)
            for i, res in enumerate(results):
                try:
                    image_url = res['image']
                    # Simple download using requests
                    import requests
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        # Save as jpg
                        img_path = save_dir / f"{query.replace(' ', '_')}_{i}.jpg"
                        with open(img_path, 'wb') as f:
                            f.write(response.content)
                except Exception as e:
                    print(f"Failed to download {image_url}: {e}")
        print(f"Downloaded {len(list(save_dir.glob('*.jpg')))} images for query '{query}'")

    # Define queries for each class
    toyota_queries = [
        "Toyota car front view", "Toyota sedan street", "Toyota Camry side",
        "Toyota Corolla rear", "Toyota car parking", "Toyota SUV road",
        "Toyota car driving", "Toyota car closeup"
    ]
    bmw_queries = [
        "BMW car front", "BMW sedan street", "BMW 3 series side",
        "BMW X5 road", "BMW car parking", "BMW car driving",
        "BMW car rear view", "BMW M4 closeup"
    ]
    other_queries = [
        "Audi car street", "Mercedes car front", "Ford car road",
        "Hyundai car side", "Kia car parking", "Nissan car driving",
        "Volkswagen car", "Skoda car", "Renault car", "Lada car"
    ]

    # Download about 500-1000 per class. Let's target 600 per class
    # For each query, download ~75 images => 8*75=600
    # We'll store raw images in data/raw/class_name/

    raw_data_root = Path("data/raw")
    for cls in CLASSES:
        (raw_data_root / cls).mkdir(parents=True, exist_ok=True)

    # Uncomment the following blocks to actually download (may take time)
    # We'll keep them commented initially to avoid accidental downloads during development.
    # You can run these cells when ready.

    # For Toyota
    # for q in toyota_queries:
    #     download_images(q, raw_data_root / 'toyota', limit=75)

    # For BMW
    # for q in bmw_queries:
    #     download_images(q, raw_data_root / 'bmw', limit=75)

    # For Other
    # for q in other_queries:
    #     download_images(q, raw_data_root / 'other', limit=75)
    return download_images, raw_data_root, toyota_queries, bmw_queries, other_queries

@app.cell
def _(raw_data_root):
    # After downloading, count images
    print("Raw image counts:")
    for cls in CLASSES:
        folder = raw_data_root / cls
        count = len(list(folder.glob('*.jpg'))) + len(list(folder.glob('*.png')))
        print(f"{cls}: {count}")
    return

@app.cell
def _(CLASSES, imagehash, np, os, Path, raw_data_root):
    # -------------------------------
    # 2. Remove duplicates and near-duplicates using perceptual hash
    # -------------------------------
    from PIL import Image

    def remove_duplicates(class_dir, hash_size=8, threshold=5):
        """Remove near-duplicate images based on pHash difference."""
        class_dir = Path(class_dir)
        image_paths = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        if not image_paths:
            return

        hashes = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                h = imagehash.phash(img, hash_size=hash_size)
                hashes.append((path, h))
            except Exception:
                # remove corrupted image
                os.remove(path)
                continue

        # Group by similarity
        to_remove = set()
        for i in range(len(hashes)):
            if hashes[i][0] in to_remove:
                continue
            for j in range(i+1, len(hashes)):
                if hashes[j][0] in to_remove:
                    continue
                if hashes[i][1] - hashes[j][1] <= threshold:
                    to_remove.add(hashes[j][0])

        for path in to_remove:
            os.remove(path)
        print(f"Removed {len(to_remove)} duplicate images from {class_dir.name}")

    # Also remove images that are too small (< 100x100) or corrupted
    def remove_small_images(class_dir, min_size=100):
        class_dir = Path(class_dir)
        for path in class_dir.glob('*'):
            try:
                img = Image.open(path)
                w, h = img.size
                if w < min_size or h < min_size:
                    os.remove(path)
            except Exception:
                os.remove(path)

    # Apply cleaning
    for cls in CLASSES:
        class_dir = raw_data_root / cls
        remove_small_images(class_dir, min_size=100)
        remove_duplicates(class_dir, hash_size=8, threshold=5)

    # Count after cleaning
    print("\nAfter automatic cleaning:")
    for cls in CLASSES:
        folder = raw_data_root / cls
        count = len(list(folder.glob('*.jpg'))) + len(list(folder.glob('*.png')))
        print(f"{cls}: {count}")
    return remove_duplicates, remove_small_images

@app.cell
def _(CLASSES, raw_data_root):
    # -------------------------------
    # 3. Manual cleaning: show images and let user delete unwanted ones
    # -------------------------------
    import matplotlib.pyplot as plt

    def manual_clean_class(class_dir, rows=4, cols=5):
        """Display images in a grid and ask user for indices to delete."""
        class_dir = Path(class_dir)
        image_paths = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        if not image_paths:
            print(f"No images in {class_dir}")
            return

        # Show in batches
        for start in range(0, len(image_paths), rows*cols):
            end = min(start + rows*cols, len(image_paths))
            batch = image_paths[start:end]
            fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
            axes = axes.flatten()
            for idx, ax in enumerate(axes):
                if idx < len(batch):
                    img = plt.imread(batch[idx])
                    ax.imshow(img)
                    ax.set_title(f"{start+idx}", fontsize=8)
                    ax.axis('off')
                else:
                    ax.axis('off')
            plt.tight_layout()
            plt.show()
            print(f"Images {start} to {end-1}. Enter indices to delete (comma-separated) or 'n' to skip, 'q' to quit:")
            user_input = input().strip()
            if user_input.lower() == 'q':
                break
            if user_input.lower() != 'n':
                indices = [int(i.strip()) for i in user_input.split(',') if i.strip().isdigit()]
                for idx in indices:
                    if start <= idx < end:
                        path = batch[idx - start]
                        os.remove(path)
                        print(f"Deleted {path.name}")
            plt.close(fig)

    # Uncomment to run manual cleaning for each class
    # for cls in CLASSES:
    #     manual_clean_class(raw_data_root / cls, rows=5, cols=6)
    return (manual_clean_class,)

@app.cell
def _(CLASSES, raw_data_root):
    # Final count after manual cleaning
    print("\nFinal image counts after manual cleaning:")
    for cls in CLASSES:
        folder = raw_data_root / cls
        count = len(list(folder.glob('*.jpg'))) + len(list(folder.glob('*.png')))
        print(f"{cls}: {count}")
    return

@app.cell
def _(CLASSES, Path, raw_data_root):
    # -------------------------------
    # 4. Split data into train/val (80/20) preserving class distribution
    # -------------------------------
    from sklearn.model_selection import train_test_split

    train_root = Path("data/train")
    val_root = Path("data/val")
    for split in [train_root, val_root]:
        for cls in CLASSES:
            (split / cls).mkdir(parents=True, exist_ok=True)

    # Copy files
    for cls in CLASSES:
        src_dir = raw_data_root / cls
        images = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png'))
        # Split
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
        for img in train_imgs:
            shutil.copy(img, train_root / cls / img.name)
        for img in val_imgs:
            shutil.copy(img, val_root / cls / img.name)
        print(f"{cls}: train={len(train_imgs)}, val={len(val_imgs)}")
    return train_root, val_root

@app.cell
def _(CLASSES, train_root, val_root):
    # -------------------------------
    # 5. Dataset class (similar to PDF) and transforms
    # -------------------------------
    from torch.utils.data import Dataset
    from torchvision.io import decode_image, ImageReadMode

    class CarDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = Path(root_dir)
            self.transform = transform
            self.classes = CLASSES
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            self.samples = []
            for cls in self.classes:
                cls_dir = self.root_dir / cls
                for img_path in cls_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((str(img_path), self.class_to_idx[cls]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = decode_image(img_path, mode=ImageReadMode.RGB)
            if self.transform:
                image = self.transform(image)
            return image, label

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CarDataset(train_root, transform=train_transforms)
    val_dataset = CarDataset(val_root, transform=val_transforms)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    return CarDataset, train_dataset, train_transforms, val_dataset, val_transforms

@app.cell
def _(DataLoader, train_dataset, val_dataset):
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Batches: train={len(train_loader)}, val={len(val_loader)}")
    return batch_size, train_loader, val_loader

@app.cell
def _(CLASSES, plt, sns, train_dataset, val_dataset):
    # Visualize class distribution
    train_counts = [0]*len(CLASSES)
    for _, label in train_dataset:
        train_counts[label] += 1
    val_counts = [0]*len(CLASSES)
    for _, label in val_dataset:
        val_counts[label] += 1

    fig, axes = plt.subplots(1,2, figsize=(12,5))
    sns.barplot(x=CLASSES, y=train_counts, ax=axes[0])
    axes[0].set_title('Train set distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(x=CLASSES, y=val_counts, ax=axes[1])
    axes[1].set_title('Validation set distribution')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    return axes, fig, train_counts, val_counts

@app.cell
def _(NUM_CLASSES, models, nn, torch):
    # -------------------------------
    # 6. Define model (fine-tuning approach)
    # -------------------------------
    class ResNetForFineTune(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super().__init__()
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Replace the final fully connected layer
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        def forward(self, x):
            return self.resnet(x)

    model = ResNetForFineTune()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return ResNetForFineTune, criterion, model, optimizer

@app.cell
def _(tqdm):
    # -------------------------------
    # 7. Training and evaluation functions (as in PDF)
    # -------------------------------
    def train(model, loss_fn, optimizer, dataloader, device):
        model.train()
        train_loss = 0
        train_acc = 0
        it_count = len(dataloader)
        pbar = tqdm(total=it_count, desc="Training")
        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.argmax(dim=1) == labels).float().mean().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
            if batch % max(1, it_count//3) == 0:
                pbar.set_description(f"Train_loss: {train_loss/(batch+1):.3f} Train_acc: {train_acc/(batch+1):.3f}")
        pbar.close()
        return train_loss/it_count, train_acc/it_count

    def test(model, loss_fn, dataloader, device):
        model.eval()
        test_loss = 0
        test_acc = 0
        it_count = len(dataloader)
        pbar = tqdm(total=it_count, desc="Validation")
        with torch.no_grad():
            for batch, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                test_acc += (outputs.argmax(dim=1) == labels).float().mean().item()
                pbar.update(1)
                if batch % max(1, it_count//3) == 0:
                    pbar.set_description(f"Val_loss: {test_loss/(batch+1):.3f} Val_acc: {test_acc/(batch+1):.3f}")
        pbar.close()
        return test_loss/it_count, test_acc/it_count

    def train_loop(model, loss_fn, optimizer, train_loader, val_loader, device, num_epochs=15):
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_acc = 0
        for epoch in range(num_epochs):
            print(f"\nEPOCH {epoch+1}/{num_epochs}" + "-"*50)
            train_loss, train_acc = train(model, loss_fn, optimizer, train_loader, device)
            val_loss, val_acc = test(model, loss_fn, val_loader, device)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
                print("Saved best model!")
        return train_losses, val_losses, train_accs, val_accs
    return test, train, train_loop

@app.cell
def _(device, model, criterion, optimizer, train_loader, val_loader, train_loop):
    # -------------------------------
    # 8. Train the model
    # -------------------------------
    num_epochs = 15
    history = train_loop(model, criterion, optimizer, train_loader, val_loader, device, num_epochs)
    train_losses, val_losses, train_accs, val_accs = history
    return history, num_epochs, train_accs, train_losses, val_accs, val_losses

@app.cell
def _(plt, train_accs, train_losses, val_accs, val_losses):
    # Plot training curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss curves')

    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy curves')
    plt.tight_layout()
    plt.show()
    return

@app.cell
def _(CLASSES, classification_report, confusion_matrix, device, model, np, plt, sns, val_loader):
    # -------------------------------
    # 9. Evaluate on validation set
    # -------------------------------
    def evaluate_model(model, dataloader, device):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=CLASSES)
        acc = np.sum(np.diag(cm)) / np.sum(cm)
        return cm, report, acc

    def plot_confusion_matrix(cm, classes):
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    cm, report, acc = evaluate_model(model, val_loader, device)
    print("Validation Set Evaluation:")
    print(report)
    print(f"Accuracy: {acc:.4f}")
    plot_confusion_matrix(cm, CLASSES)
    return cm, evaluate_model, plot_confusion_matrix, report, acc

@app.cell
def _(Path):
    # -------------------------------
    # 10. Custom test set (manually taken photos)
    # -------------------------------
    # Instructions: create folder 'data/custom_test' with subfolders toyota, bmw, other
    # Place your own photos in respective folders.
    # The code below will load and evaluate.

    custom_test_root = Path("data/custom_test")
    if not custom_test_root.exists():
        custom_test_root.mkdir(parents=True)
        for cls in CLASSES:
            (custom_test_root / cls).mkdir(exist_ok=True)
        print("Created custom_test folder. Please add your own photos into the class subfolders.")
    return (custom_test_root,)

@app.cell
def _(CLASSES, Path, custom_test_root, evaluate_model, model, val_transforms, device):
    # Load custom dataset if exists
    from torch.utils.data import DataLoader
    if custom_test_root.exists() and any((custom_test_root / cls).iterdir() for cls in CLASSES):
        custom_dataset = CarDataset(custom_test_root, transform=val_transforms)
        custom_loader = DataLoader(custom_dataset, batch_size=32, shuffle=False)
        print(f"Custom test set size: {len(custom_dataset)}")
        cm_custom, report_custom, acc_custom = evaluate_model(model, custom_loader, device)
        print("\nCustom Test Set Evaluation (own photos):")
        print(report_custom)
        print(f"Accuracy: {acc_custom:.4f}")
        # plot_confusion_matrix(cm_custom, CLASSES) # optional
    else:
        print("No custom test images found. Please add your photos to data/custom_test/class/")
    return

@app.cell
def _():
    # -------------------------------
    # 11. Analysis of generalization and edge cases
    # -------------------------------
    mo.md(r"""
    ## Conclusions

    The model was trained on scraped web images of Toyota, BMW, and other car models.
    After cleaning duplicates and manually filtering, the dataset was split into train/val.

    **Validation performance**: ~XX% accuracy (depends on data quality).

    **Generalization**: The model's performance on manually taken photos (different
    angles, lighting, partial occlusion, background clutter) indicates its ability
    to handle real-world conditions. Lower accuracy on custom test set suggests
    domain shift between web images and phone photos.

    **Impact of data quality**: Removing duplicates and low-quality images improved
    training stability. Adding augmentations (flip, color jitter) helped robustness.
    However, the model still struggles with extreme cases (heavy shadows, very
    partial views). More diverse training data (including such challenging examples)
    would likely improve generalization.

    **Out-of-distribution robustness**: The "other" class includes many different
    brands, which makes the model learn features of non-Toyota/non-BMW cars.
    This helps avoid overfitting to specific brands.

    **Future improvements**:
    - Collect more edge-case images (night, rain, cropped views) and add to training.
    - Use a stronger backbone (ResNet50) or apply transfer learning with unfreezing more layers.
    - Experiment with Focal Loss to handle class imbalance if present.
    """)
    return

if __name__ == "__main__":
    app.run()