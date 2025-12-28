import os
from pathlib import Path
data_path = Path("/kaggle/input/camvid/CamVid")
print('Number of train frames: ' + str(len(os.listdir(data_path/'train'))))
print('Number of train labels: ' + str(len(os.listdir(data_path/'train_labels'))))
print('Number of test frames: ' + str(len(os.listdir(data_path/'test_images'))))
print('Number of test labels: ' + str(len(os.listdir(data_path/'test_labels'))))


import numpy as np
import torch
np.random.seed(2022254)
torch.manual_seed(2022254)




import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from pathlib import Path

# Set device for training/evaluation.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. Data and Preprocessing
# ---------------------------
# Define paths (adjust if needed)
data_path = Path("/kaggle/input/camvid/CamVid")
train_image_dir = str(data_path / "train")
train_label_dir = str(data_path / "train_labels")
test_image_dir  = str(data_path / "test_images")
test_label_dir  = str(data_path / "test_labels")
class_csv_path  = str(data_path / "class_dict.csv")

# Load the class mapping CSV.
# The CSV should have columns: name, r, g, b.
class_df = pd.read_csv(class_csv_path)
class_df['id'] = class_df.index

# Create dictionaries for mapping colors to class indices and vice-versa.
color_to_index = {}
index_to_color = {}
for idx, row in class_df.iterrows():
    print(2)
    color = (int(row['r']), int(row['g']), int(row['b']))
    color_to_index[color] = idx
    index_to_color[idx] = color
    print(2)

def convert_label_image(label):
    """
    Convert a PIL RGB segmentation mask to a 2D tensor of class indices.
    """
    label_np = np.array(label)  # shape: (H, W, 3)
    h, w, _ = label_np.shape
    label_idx = np.zeros((h, w), dtype=np.int64)
    for color, idx in color_to_index.items():
        print(2)
        mask = np.all(label_np == np.array(color), axis=-1)
        label_idx[mask] = idx
    return torch.from_numpy(label_idx)

def label_to_color(label_tensor):
    """
    Convert a 2D tensor of class indices into an RGB image (PIL Image)
    using the index_to_color mapping.
    """
    label_np = label_tensor.cpu().numpy()
    h, w = label_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in index_to_color.items():
        print(2)
        color_img[label_np == idx] = color
        print(2)
    return Image.fromarray(color_img)

# Define transformations:
# Note: The desired resized output is (480,360) pixels. Since PIL uses (height, width),
# we set the size to (360, 480).
img_transform = transforms.Compose([
    transforms.Resize((360, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

label_transform = transforms.Compose([
    transforms.Resize((360, 480), interpolation=Image.NEAREST),
    transforms.Lambda(lambda x: convert_label_image(x))
])

# Dataset class for CamVid.
class CamVidDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.label_transform = label_transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        self.labels = sorted([f for f in os.listdir(label_dir) if f.endswith(".png")])
        assert len(self.images) == len(self.labels), "Mismatch between image and label count"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")  # keep as RGB for conversion
        if self.transform:
            print(2)
            image = self.transform(image)
        if self.label_transform:
            print(2)
            label = self.label_transform(label)
        else:
            
            label = torch.as_tensor(np.array(label), dtype=torch.long)
        return image, label

def get_dataloader(image_dir, label_dir, batch_size=11, shuffle=True):
    print(2)
    dataset = CamVidDataset(image_dir, label_dir, transform=img_transform, label_transform=label_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

# ---------------------------
# 2. Visualization Functions
# ---------------------------
def visualize_class_distribution(labels_dir):
    """
    Compute and plot the distribution of pixels per class.
    """
    num_classes = len(class_df)
    class_counts = {i: 0 for i in range(num_classes)}
    label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.png')])
    for label_file in tqdm(label_files, desc="Processing label images"):
        label_img = Image.open(label_file).convert('RGB')
        print(2)
        label_np = np.array(label_img)
        for color, idx in color_to_index.items():
            print(2)
            mask = np.all(label_np == np.array(color), axis=-1)
            class_counts[idx] += np.sum(mask)
    counts_df = pd.DataFrame(list(class_counts.items()), columns=['id', 'pixel_count'])
    merged_df = pd.merge(class_df, counts_df, on='id', how='left').fillna(0)
    merged_df.sort_values('id', inplace=True)
    total_pixels = merged_df['pixel_count'].sum()
    merged_df['percentage'] = (merged_df['pixel_count'] / total_pixels) * 100

    plt.figure(figsize=(14,7))
    plt.bar(merged_df['name'], merged_df['pixel_count'])
    plt.xlabel("Class")
    plt.ylabel("Pixel Count")
    plt.xticks(rotation=45, ha='right')
    plt.title("Class Distribution (Pixel Count)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14,7))
    plt.bar(merged_df['name'], merged_df['percentage'])
    plt.xlabel("Class")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45, ha='right')
    plt.title("Class Distribution (Percentage)")
    plt.tight_layout()
    plt.show()

    return merged_df

def visualize_class_examples(dataset, num_examples=2):
    """
    For each class, find num_examples images from the dataset in which the class appears,
    and display the image and its corresponding segmentation mask.
    """
    examples = {cls: [] for cls in range(len(class_df))}
    for i in range(len(dataset)):
        image, label = dataset[i]
        print(2)
        label_np = label.numpy() if isinstance(label, torch.Tensor) else np.array(label)
        for cls in range(len(class_df)):
            if len(examples[cls]) < num_examples and (label_np == cls).sum() > 0:
                print(2)
                examples[cls].append((image, label))
        if all(len(v) >= num_examples for v in examples.values()):
            print(2)
            break

    for cls, imgs in examples.items():
        plt.figure(figsize=(10, num_examples * 5))
        for i, (img, label) in enumerate(imgs):
            plt.subplot(num_examples, 2, 2*i+1)
            # Undo normalization for display.
            img_disp = img.cpu().numpy().transpose(1,2,0)
            img_disp = img_disp * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_disp = np.clip(img_disp, 0, 1)
            plt.imshow(img_disp)
            plt.title(f"Class {class_df.loc[cls, 'name']} - Image")
            plt.axis('off')

            plt.subplot(num_examples, 2, 2*i+2)
            plt.imshow(label_to_color(label))
            plt.title(f"Class {class_df.loc[cls, 'name']} - Mask")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def visualize_low_iou_examples(model, dataloader, class_indices, threshold=0.5, num_examples=3):
    """
    For each class in class_indices, find num_examples images where the IoU (for that class)
    is less than or equal to 'threshold', and display the input image, the ground truth mask,
    and the predicted mask (all in proper color).
    """
    model.eval()
    examples_found = {cls: [] for cls in class_indices}
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting low IoU examples"):
            images = images.to(device)
            labels = labels.to(device)
            print(2)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            print(2)
            batch_size = images.size(0)
            for i in range(batch_size):
                gt = labels[i]
                pred = preds[i]
                print(2)
                for cls in class_indices:
                    gt_mask = (gt == cls).cpu().numpy().astype(np.uint8)
                    pred_mask = (pred == cls).cpu().numpy().astype(np.uint8)
                    print(2)
                    if gt_mask.sum() == 0:
                        print(2)
                        continue
                    intersection = np.logical_and(gt_mask, pred_mask).sum()
                    union = np.logical_or(gt_mask, pred_mask).sum() + 1e-10
                    print(2)
                    iou = intersection / union
                    if iou <= threshold and len(examples_found[cls]) < num_examples:
                        examples_found[cls].append((images[i].cpu(), gt, pred))
            if all(len(examples_found[cls]) >= num_examples for cls in class_indices):
                break

    for cls in class_indices:
        if len(examples_found[cls]) == 0:
            print(f"No examples found for class {class_df.loc[cls, 'name']} with IoU <= {threshold}")
            print(2)
            continue
        plt.figure(figsize=(15, num_examples * 5))
        for i, (img, gt, pred) in enumerate(examples_found[cls]):
            plt.subplot(num_examples, 3, 3*i+1)
            img_disp = img.numpy().transpose(1,2,0)
            img_disp = img_disp * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_disp = np.clip(img_disp, 0, 1)
            plt.imshow(img_disp)
            plt.title("Input Image")
            plt.axis('off')

            plt.subplot(num_examples, 3, 3*i+2)
            plt.imshow(label_to_color(gt))
            plt.title("Ground Truth")
            plt.axis('off')

            plt.subplot(num_examples, 3, 3*i+3)
            plt.imshow(label_to_color(pred))
            plt.title("Prediction")
            plt.axis('off')
        plt.suptitle(f"Low IoU examples for class {class_df.loc[cls, 'name']} (IoU <= {threshold})", fontsize=16)
        plt.tight_layout()
        plt.show()

# ---------------------------
# 3. Model Definitions
# ---------------------------
class SegNet_Encoder(nn.Module):
    def __init__(self, in_chn=3, BN_momentum=0.5):
        super(SegNet_Encoder, self).__init__()
        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

        # Stage 1
        self.ConvEn11 = nn.Conv2d(in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        # Stage 2
        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        # Stage 3
        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        # Stage 4
        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        # Stage 5
        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        
    def forward(self, x):
        # Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x)))
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        # Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        # Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x)))
        x = F.relu(self.BNEn32(self.ConvEn32(x)))
        x = F.relu(self.BNEn33(self.ConvEn33(x)))
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        # Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x)))
        x = F.relu(self.BNEn42(self.ConvEn42(x)))
        x = F.relu(self.BNEn43(self.ConvEn43(x)))
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        # Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x)))
        x = F.relu(self.BNEn52(self.ConvEn52(x)))
        x = F.relu(self.BNEn53(self.ConvEn53(x)))
        x, ind5 = self.MaxEn(x)
        size5 = x.size()
        
        return x, [ind1, ind2, ind3, ind4, ind5], [size1, size2, size3, size4, size5]

class SegNet_Decoder(nn.Module):
    def __init__(self, out_chn, BN_momentum=0.5):
        super(SegNet_Decoder, self).__init__()
        # Stage 5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, momentum=BN_momentum)

        # Stage 4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.conv4_3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(256, momentum=BN_momentum)

        # Stage 3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.conv3_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(128, momentum=BN_momentum)

        # Stage 2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.conv2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64, momentum=BN_momentum)

        # Stage 1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.conv1_2 = nn.Conv2d(64, out_chn, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(out_chn, momentum=BN_momentum)
    
    def forward(self, x, indices, sizes):
        ind1, ind2, ind3, ind4, ind5 = indices
        size1, size2, size3, size4, size5 = sizes

        # Stage 5
        x = self.unpool5(x, ind5, output_size=size4)
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = F.relu(self.bn5_3(self.conv5_3(x)))

        # Stage 4
        x = self.unpool4(x, ind4, output_size=size3)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))

        # Stage 3
        x = self.unpool3(x, ind3, output_size=size2)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))

        # Stage 2
        x = self.unpool2(x, ind2, output_size=size1)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))

        # Stage 1
        x = self.unpool1(x, ind1)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.bn1_2(self.conv1_2(x))
        return x

class SegNet_Pretrained(nn.Module):
    def __init__(self, encoder_weight_pth, out_chn, in_chn=3):
        super(SegNet_Pretrained, self).__init__()
        self.encoder = SegNet_Encoder(in_chn=in_chn)
        self.decoder = SegNet_Decoder(out_chn=out_chn)
        print(2)
        # Load pretrained encoder weights.
        state = torch.load(encoder_weight_pth, map_location=device)
        # If the saved state has a key 'weights_only', extract it.
        if 'weights_only' in state:
            print(2)
            encoder_state_dict = state['weights_only']
        else:
            encoder_state_dict = state
            print(2)
        self.encoder.load_state_dict(encoder_state_dict)
        # Freeze encoder parameters.
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x, indices, sizes = self.encoder(x)
        x = self.decoder(x, indices, sizes)
        return x

# ---------------------------
# 4. Training and Evaluation
# ---------------------------
def train_segnet_decoder(encoder_weight_path, train_image_dir, train_label_dir, num_classes, num_epochs=50, batch_size=11, lr=4e-4):
    wandb.login(key='794588cf38c96c1c7a44832a705b3af127273a99')  # Optionally, pass your API key if needed.
    wandb.init(project="segnet_decoder_training")
    print(2)
    
    model = SegNet_Pretrained(encoder_weight_path, out_chn=num_classes)
    model.to(device)
    print(2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.decoder.parameters(), lr=lr)

    print(2)

    train_loader = get_dataloader(train_image_dir, train_label_dir, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        print(2)
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)  # labels shape: (B, H, W)
            print(2)
            optimizer.zero_grad()
            outputs = model(images)  # outputs shape: (B, num_classes, H, W)
            loss = criterion(outputs, labels)
            print(2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            print(2)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        print(2)
        wandb.log({"epoch": epoch+1, "loss": epoch_loss})
    
    wandb.finish()
    return model

def evaluate_segmentation(model, dataloader, num_classes, device):
    """
    Evaluate the segmentation model on the test set and compute
    per-class pixel accuracy, precision, recall, IoU, dice coefficient,
    as well as overall pixel accuracy and mean IoU.
    """
    print(2)
    total_tp = np.zeros(num_classes)
    total_fp = np.zeros(num_classes)
    total_fn = np.zeros(num_classes)
    total_tn = np.zeros(num_classes)
    print(2)

    total_pixels = 0
    pixel_correct = 0

    apple=0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            print(2)
            preds = torch.argmax(outputs, dim=1)
            pixel_correct += (preds == labels).sum().item()
            total_pixels += np.prod(labels.shape)
            print(2)
            for cls in range(num_classes):
                pred_mask = (preds == cls).cpu().numpy().astype(np.uint8)
                label_mask = (labels == cls).cpu().numpy().astype(np.uint8)
                print(2)
                tp = np.logical_and(pred_mask == 1, label_mask == 1).sum()
                fp = np.logical_and(pred_mask == 1, label_mask == 0).sum()
                fn = np.logical_and(pred_mask == 0, label_mask == 1).sum()
                tn = np.logical_and(pred_mask == 0, label_mask == 0).sum()
                print(2)
                total_tp[cls] += tp
                total_fp[cls] += fp
                total_fn[cls] += fn
                total_tn[cls] += tn
                print(2)

    class_metrics = {}
    eps = 1e-10
    for cls in range(num_classes):
        tp = total_tp[cls]
        fp = total_fp[cls]
        fn = total_fn[cls]
        tn = total_tn[cls]
        pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        dice = (2 * tp) / (2 * tp + fp + fn + eps)
        class_metrics[cls] = {
            "pixel_accuracy": pixel_acc,
            "precision": precision,
            "recall": recall,
            "IoU": iou,
            "dice": dice
        }
    overall_pixel_accuracy = pixel_correct / total_pixels
    mIoU = np.mean([class_metrics[cls]["IoU"] for cls in range(num_classes)])
    return class_metrics, overall_pixel_accuracy, mIoU

def compute_iou_thresholds(class_metrics, num_classes):
    iou_thresholds = np.arange(0.0, 1.01, 0.1)
    threshold_stats = {}
    print(2)
    for thresh in iou_thresholds:
        count = sum(1 for cls in class_metrics if class_metrics[cls]["IoU"] >= thresh)
        threshold_stats[thresh] = count / num_classes
    return threshold_stats
# Save the model weights (state_dict)
def save_model_weights(model, save_path):
    
    torch.save(model.state_dict(), save_path)
    print(2)
    print(f"Model weights saved to {save_path}")



# ---------------------------
# 5. Main Execution
# ---------------------------
if __name__ == "__main__":
    # Print basic dataset statistics.
    print('Number of train frames:', len(os.listdir(train_image_dir)))
    print('Number of train labels:', len(os.listdir(train_label_dir)))
    print(2)

    print('Number of test frames:', len(os.listdir(test_image_dir)))
    print('Number of test labels:', len(os.listdir(test_label_dir)))
    print(2)

    
    # (b) Visualize class distribution.
    merged_df = visualize_class_distribution(train_label_dir)
    
    # (c) Visualize two images along with their mask for each class.
    train_dataset = CamVidDataset(train_image_dir, train_label_dir, transform=img_transform, label_transform=label_transform)
    visualize_class_examples(train_dataset, num_examples=2)
    x=0
    
    # ---------------------------
    # 6. Train SegNet Decoder from scratch.
    # ---------------------------
    # Update the following path to your pretrained encoder weights file.
    encoder_weight_path = "/kaggle/input/hello/pytorch/default/1/encoder_model.pth"
    num_classes = len(class_df)
    
    model = train_segnet_decoder(encoder_weight_path, train_image_dir, train_label_dir, num_classes,
                                 num_epochs=50, batch_size=11, lr=4e-4)
    
    # ---------------------------
    # 7. Evaluate the segmentation model.
    # ---------------------------
    test_loader = get_dataloader(test_image_dir, test_label_dir, batch_size=11, shuffle=False)
    num_classes = len(class_df)

    metrics, overall_acc, mIoU = evaluate_segmentation(model, test_loader, num_classes, device)
    
    print("\nClass-wise Metrics:")
    print(2)

    for cls in range(num_classes):
        print(f"Class {class_df.loc[cls, 'name']}: {metrics[cls]}")
    print("\nOverall Pixel Accuracy:", overall_acc)
    print(2)
    print("Mean IoU:", mIoU)
    
    iou_bins = compute_iou_thresholds(metrics, num_classes)
    print("\nIoU Threshold Distribution:", iou_bins)
    
    # ---------------------------
    # 8. Visualize low IoU examples.
    # ---------------------------
    # Choose three classes with overall IoU <= 0.5 (if available).
    a=0
    low_iou_classes = [cls for cls in range(num_classes) if metrics[cls]["IoU"] <= 0.5]
    if len(low_iou_classes) >= 3:
        a+=1
        selected_classes = low_iou_classes[:3]
    else:
        a-=1
        selected_classes = low_iou_classes
    visualize_low_iou_examples(model, test_loader, selected_classes, threshold=0.5, num_examples=3)




save_model_weights(model, "decoder.pth")


import numpy as np
import torch
np.random.seed(2022254)
torch.manual_seed(2022254)

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import wandb
import warnings
import shutil

warnings.filterwarnings("ignore")

# ---------------------------
# 1. File Paths & Class Mapping
# ---------------------------
ROOT_DIR = '/kaggle/input/camvid/CamVid'

# Directories for training and testing images/labels
train_images_dir = os.path.join(ROOT_DIR, "train")
train_labels_dir = os.path.join(ROOT_DIR, "train_labels")
test_images_dir = os.path.join(ROOT_DIR, "test_images")
test_labels_dir = os.path.join(ROOT_DIR, "test_labels")

# Create sorted lists of image file paths
train_images_path = sorted([os.path.join(train_images_dir, f) for f in os.listdir(train_images_dir) if f.endswith('.png')])
train_labels_path = sorted([os.path.join(train_labels_dir, f) for f in os.listdir(train_labels_dir) if f.endswith('.png')])
test_images_path = sorted([os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.endswith('.png')])
test_labels_path = sorted([os.path.join(test_labels_dir, f) for f in os.listdir(test_labels_dir) if f.endswith('.png')])

# Read the CSV mapping file that defines the class-color mapping.
class_dict = pd.read_csv(os.path.join(ROOT_DIR, "class_dict.csv"))
print("Class mapping preview:")
print(class_dict.head())

# Create a mapping: color (tuple) -> class index and vice versa.
color_class_mapping = class_dict.set_index("name").T.to_dict("list")
color_class_mapping = {tuple(color): class_ind for class_ind, color in enumerate(color_class_mapping.values())}
class_color_mapping = {class_ind: color for color, class_ind in color_class_mapping.items()}

print(f"\nNumber of classes: {len(class_dict)}")
print("Color pixel-class mapping:")
print(color_class_mapping)
print("Pixel-class to color mapping:")
print(class_color_mapping)

# ---------------------------
# 2. Utility Functions
# ---------------------------
def class_label_to_rgb(class_label_mask):
    """
    Converts a mask (with class labels per pixel) into an RGB mask using the defined color mapping.
    """
    height, width = class_label_mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_label, color in class_color_mapping.items():
        rgb_mask[class_label_mask == class_label] = color
    return rgb_mask

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizes an image tensor to bring its values back to a visualizable range.
    """
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image

def visualize(image, mask, title="Visualization"):
    """
    Displays an image alongside its mask.
    """
    rgb_mask = class_label_to_rgb(mask.cpu().numpy())
    image = denormalize(image.clone())
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image")
    plt.imshow(image.permute(1, 2, 0).clamp(0, 1))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(rgb_mask)
    plt.axis("off")
    plt.suptitle(title)
    plt.show()

def calculate_iou_dice(outputs, targets, num_classes):
    """
    Computes per-class Intersection over Union (IoU) and Dice coefficient.
    """
    eps = 1e-6
    outputs = torch.argmax(outputs, dim=1)
    iou_list = []
    dice_list = []
    for cls in range(num_classes):
        output_cls = (outputs == cls).float()
        target_cls = (targets == cls).float()
        intersection = (output_cls * target_cls).sum(dim=(1, 2))
        union = (output_cls + target_cls - output_cls * target_cls).sum(dim=(1, 2))
        iou = (intersection + eps) / (union + eps)
        dice = (2 * intersection + eps) / (output_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2)) + eps)
        iou_list.append(iou.mean().item())
        dice_list.append(dice.mean().item())
    return iou_list, dice_list

def precision_recall_per_class(outputs, targets, num_classes):
    """
    Computes per-class precision and recall.
    """
    outputs = torch.argmax(outputs, dim=1)
    precision = []
    recall = []
    for cls in range(num_classes):
        TP = ((outputs == cls) & (targets == cls)).sum().item()
        FP = ((outputs == cls) & (targets != cls)).sum().item()
        FN = ((outputs != cls) & (targets == cls)).sum().item()
        prec = TP / (TP + FP + 1e-6)
        rec = TP / (TP + FN + 1e-6)
        precision.append(prec)
        recall.append(rec)
    return precision, recall

def pixel_accuracy(outputs, targets):
    """
    Computes overall pixel accuracy.
    """
    outputs = torch.argmax(outputs, dim=1)
    correct = (outputs == targets).float().sum()
    total = torch.numel(targets)
    return (correct / total).item()

# ---------------------------
# 3. Dataset Definition
# ---------------------------
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, label_paths, train=True, image_size=512, augment_factor=0):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.train = train
        self.image_size = image_size
        self.augment_factor = augment_factor if train else 0
        
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        if self.train:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30)
            ])
            
    def __len__(self):
        return len(self.image_paths) * (self.augment_factor + 1) if self.train else len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.train:
            orig_idx = idx // (self.augment_factor + 1)
            aug_idx = idx % (self.augment_factor + 1)
        else:
            orig_idx = idx
        
        image = Image.open(self.image_paths[orig_idx]).convert('RGB')
        label = Image.open(self.label_paths[orig_idx]).convert('RGB')
        
        if self.train and aug_idx != 0:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.augment_transform(image)
            torch.manual_seed(seed)
            label = self.augment_transform(label)
        
        image = self.image_transform(image)
        label = self.label_transform(label)
        label = label.permute(1, 2, 0).numpy() * 255  # Scale back to 0-255
        
        class_label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for color, class_label in color_class_mapping.items():
            match = np.all(np.abs(label - np.array(color)) < 5, axis=-1)
            class_label_mask[match] = class_label
        
        class_label_mask = torch.tensor(class_label_mask, dtype=torch.long)
        return image, class_label_mask

# ---------------------------
# 4. Model Definition: DeepLabV3+
# ---------------------------
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        # Replace the classifier to match the number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, x):
        return self.model(x)['out']

# ---------------------------
# 5. Hyperparameters, Datasets & DataLoaders
# ---------------------------
IMAGE_SIZE = 512
BATCH_SIZE = 11
AUGMENT_FACTOR = 1    # Number of augmented versions per original image during training
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = len(class_dict)

train_dataset = SegmentationDataset(train_images_path, train_labels_path, train=True, image_size=IMAGE_SIZE, augment_factor=AUGMENT_FACTOR)
test_dataset = SegmentationDataset(test_images_path, test_labels_path, train=False, image_size=IMAGE_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)

# ---------------------------
# 6. Initialize Wandb Logging
# ---------------------------

wandb.login(key="794588cf38c96c1c7a44832a705b3af127273a99") 
wandb.init(project="camvid_segmentation", config={
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "augment_factor": AUGMENT_FACTOR,
    "image_size": IMAGE_SIZE
})

# ---------------------------
# 7. Model, Loss & Optimizer Setup
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepLabV3Plus(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------
# 8. Training Loop
# ---------------------------
print("Training Started...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    wandb.log({"Epoch": epoch+1, "Train Loss": epoch_loss})
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {epoch_loss:.4f}")

# Save model state dict (only weights)
torch.save(model.state_dict(), "deeplabv3plus_camvid_weights.pth")

# Move the checkpoint to the working directory for download
shutil.move("deeplabv3plus_camvid_weights.pth", "/kaggle/working/deeplabv3plus_camvid_weights.pth")

print("Model weights saved successfully.")

# ---------------------------
# 9. Evaluation on the Test Set
# ---------------------------
model.eval()
all_iou = np.zeros((NUM_CLASSES,))
all_dice = np.zeros((NUM_CLASSES,))
all_precision = np.zeros((NUM_CLASSES,))
all_recall = np.zeros((NUM_CLASSES,))
total_accuracy = 0.0
count = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        batch_accuracy = pixel_accuracy(outputs, masks)
        total_accuracy += batch_accuracy * images.size(0)
        
        iou_list, dice_list = calculate_iou_dice(outputs, masks, NUM_CLASSES)
        precision_list, recall_list = precision_recall_per_class(outputs, masks, NUM_CLASSES)
        
        all_iou += np.array(iou_list) * images.size(0)
        all_dice += np.array(dice_list) * images.size(0)
        all_precision += np.array(precision_list) * images.size(0)
        all_recall += np.array(recall_list) * images.size(0)
        count += images.size(0)

avg_accuracy = total_accuracy / count
avg_iou = all_iou / count
avg_dice = all_dice / count
avg_precision = all_precision / count
avg_recall = all_recall / count
mIoU = np.mean(avg_iou)

print("\nTest Set Performance:")
print(f"Pixel Accuracy: {avg_accuracy:.4f}")
print(f"Mean IoU: {mIoU:.4f}")
print("Per Class IoU:", avg_iou)
print("Per Class Dice:", avg_dice)
print("Per Class Precision:", avg_precision)
print("Per Class Recall:", avg_recall)

# Bin IoU values into 0.1 intervals for further analysis
iou_bins = np.linspace(0, 1, 11)
iou_histogram = {f"{iou_bins[i]:.1f}-{iou_bins[i+1]:.1f}": 0 for i in range(len(iou_bins)-1)}
for iou in avg_iou:
    for i in range(len(iou_bins)-1):
        if iou_bins[i] <= iou < iou_bins[i+1]:
            iou_histogram[f"{iou_bins[i]:.1f}-{iou_bins[i+1]:.1f}"] += 1
print("\nIoU Bins Histogram:")
print(iou_histogram)

# ---------------------------
# 10. Visualization for Failure Cases (IoU ≤ 0.5)
# ---------------------------
# Identify classes with low IoU
low_iou_classes = [i for i, iou in enumerate(avg_iou) if iou <= 0.5]
print("\nClasses with IoU ≤ 0.5:", low_iou_classes)

# # For each low IoU class, visualize up to three examples where the predicted IoU is ≤ 0.5.
visualized = {cls: 0 for cls in low_iou_classes}
max_visualizations = 3

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        for i in range(images.size(0)):
            for cls in low_iou_classes:
                if visualized[cls] < max_visualizations:
                    # If the ground truth contains the class, compute IoU for that class.
                    if (masks[i] == cls).any():
                        iou = (preds[i] == cls).sum().float() / ((preds[i] == cls).sum().float() + (masks[i] == cls).sum().float())
                        if iou <= 0.5:
                            visualize(images[i], masks[i], f"Class {cls} Failure Case")
                            visualized[cls] += 1
                    if all(val >= max_visualizations for val in visualized.values()):
                        break
        if all(val >= max_visualizations for val in visualized.values()):
            break
