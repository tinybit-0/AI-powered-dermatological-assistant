import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import time
import copy
from PIL import Image

DATA_DIR = 'data'
NUM_CLASSES = 7
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
lesion_type_dict = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions',
    'mel': 'Melanoma'
}


def prepare_data_from_npz(npz_file_path, data_dir):
   
    if os.path.exists(os.path.join(data_dir, 'train')) and \
       os.path.exists(os.path.join(data_dir, 'val')) and \
       os.path.exists(os.path.join(data_dir, 'test')):
        print("Data directories already exist. Skipping NPZ preparation.")
        return

    print("Loading data from NPZ file...")
    data = np.load(npz_file_path)
    
    splits = {
        'train': (data['train_images'], data['train_labels']),
        'val': (data['val_images'], data['val_labels']),
        'test': (data['test_images'], data['test_labels'])
    }
    
    for split_name, (split_images, split_labels) in splits.items():
        split_path = os.path.join(data_dir, split_name)
        
        for class_idx in range(NUM_CLASSES):
            class_dir = os.path.join(split_path, str(class_idx))
            os.makedirs(class_dir, exist_ok=True)
            
        print(f"Saving {split_name} data to disk...")
        for i, (image, label) in enumerate(zip(split_images, split_labels)):
            label_int = int(label.item()) if isinstance(label, np.ndarray) and label.ndim > 0 else int(label)
            
            if image.ndim == 2:  # Grayscale image
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 1: # Single channel image
                image = np.concatenate([image] * 3, axis=-1)

            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            img_pil = Image.fromarray(image)
            img_pil.save(os.path.join(split_path, str(label_int), f"{i}.png"))

    print("Data preparation complete.")


if __name__ == '__main__':
    prepare_data_from_npz('Data.npz', DATA_DIR)

    print("Loading and preprocessing data...")
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                 shuffle=True if x == 'train' else False, num_workers=4)
                   for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Class names: {class_names}")

    print("Setting up the model...")

    model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print("Starting training...")

    def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        model.load_state_dict(best_model_wts)
        return model

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

    print("Evaluating the model on the test set...")

    def evaluate_model(model, dataloader, class_names):
        model.eval() 
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)

        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="left")
        ax.set_yticklabels(class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

    evaluate_model(model_ft, dataloaders['test'], class_names)

    print("Saving the trained model...")
    torch.save(model_ft.state_dict(), 'skin_lesion_classifier.pth')
    print("Model saved as skin_lesion_classifier.pth")

    def predict_image(model, image_path, class_names, device):
        model.eval() 
        transform = data_transforms['test'] 

        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            confidence, predicted_class_idx = torch.max(probabilities, 0)

        predicted_class_name = class_names[predicted_class_idx.item()]
        print(f"\nPrediction for {image_path}:")
        print(f"Predicted Class: {predicted_class_name}")
        print(f"Confidence: {confidence.item():.4f}")

        top_p, top_class = probabilities.topk(3, dim=0)
        print("\nTop 3 Predictions:")
        for i in range(top_p.size(0)):
            print(f"{class_names[top_class[i].item()]}: {top_p[i].item():.4f}")
