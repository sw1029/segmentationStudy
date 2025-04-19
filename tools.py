import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
from pycocotools.coco import COCO
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class customDataset(Dataset):
    def __init__(self, imgList, maskList):
        self.imgList = imgList
        self.maskList = maskList
    def __len__(self):
        return len(self.imgList)
    def __getitem__(self, idx):
        img = self.imgList[idx]
        mask = self.maskList[idx]
        return img, mask

def dataMaker():
    sample_annotation_path = "C:/Users/USER/Downloads/sample/annotations/instances_default.json"
    sample_image_path = "C:/Users/USER/Downloads/sample/images/default"
    # COCO annotation 로드
    coco = COCO(sample_annotation_path)

    # 이미지 전처리를 위한 transform (1024x1024로 resize)
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])

    img_list = []
    mask_list = []

    img_ids = coco.getImgIds()
    NUM_CLASSES = 91  # 필요 시 80 또는 133으로 조정

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(sample_image_path, file_name)
        image = Image.open(img_path).convert('RGB')
        image = transform(image)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        if len(ann_ids) == 0:
            continue
        anns = coco.loadAnns(ann_ids)

        height, width = img_info['height'], img_info['width']
        class_mask = np.zeros((height, width), dtype=np.uint8)  # class ID 저장용

        for ann in anns:
            class_id = ann['category_id']  # COCO class ID
            mask_ann = coco.annToMask(ann)  # [H, W]
            class_mask = np.where((mask_ann == 1) & (class_mask == 0), class_id, class_mask)

        # → torch.tensor로 변환 (shape: [H, W], 값: class_id)
        mask_tensor = torch.tensor(class_mask, dtype=torch.long)

        # Resize to 1024x1024
        if height != 1024 or width != 1024:
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).float(),  # [1, H, W] → float for interpolate
                size=(1024, 1024),
                mode='nearest'
            ).squeeze(0).long()  # → [1024, 1024], long 다시 복원

        # One-hot encoding → [H, W, C]
        mask_tensor = F.one_hot(mask_tensor, num_classes=NUM_CLASSES)  # [H, W, C]
        mask_tensor = mask_tensor.permute(2, 0, 1).float()  # [C, H, W]

        img_list.append(image)
        mask_list.append(mask_tensor)
    # Dataset 생성
    dataset = customDataset(img_list, mask_list)

    # Train/Test split
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def model_train(model, device, criterion, optimizer, scheduler, dataloader_train, dataloader_valid, epochs=300):
    model.to(device)
    valid_loss_list = []
    valid_acc_list = []

    # 초기 validation
    val_loss, val_accuracy = model_valid(model, device, criterion, dataloader_valid)
    valid_loss_list.append(val_loss)
    valid_acc_list.append(val_accuracy)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}")

        for batch in loop:
            img, label = batch
            img = img.to(device)  # (B, 3, 1024, 1024)
            label = label.to(device)  # [B,C,H,W]

            optimizer.zero_grad()
            pred = model(img)#[B,C,H,W] 형태
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(dataloader_train)
        print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.6f}")

        # validation
        val_loss, val_accuracy = model_valid(model, device, criterion, dataloader_valid)
        scheduler.step(val_loss)
        valid_loss_list.append(val_loss)
        valid_acc_list.append(val_accuracy)

        # 10 epoch마다 임시 저장
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model/model_param_temp/model_{epoch}.pth")
            plot_save(valid_loss_list, valid_acc_list, show_plot=False)

    # 학습 종료 후 최종 plot
    plot_save(valid_loss_list, valid_acc_list, show_plot=True)


def model_valid(model, device, criterion, dataloader_valid):
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    loop = tqdm(dataloader_valid, desc="Validation")
    with torch.no_grad():
        for batch in loop:
            img, label = batch
            img = img.to(device)
            label = label.to(device)

            pred = model(img)  # [B,C,H,W]
            loss = criterion(pred, label)
            total_loss += loss.item()

            pred_binary = (pred > 0.5).float()
            total_correct += (pred_binary == label).sum().item()
            total_pixels += label.numel()

    avg_loss = total_loss / len(dataloader_valid)
    accuracy = total_correct / total_pixels
    print(f"Validation Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")

    model.train()
    return avg_loss, accuracy


def plot_save(loss_list, acc_list, show_plot=False):
    epochs = range(1, len(loss_list) + 1)
    plt.figure()
    plt.plot(epochs, loss_list, label='Validation Loss')
    plt.plot(epochs, acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Validation Loss and Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('csv/val_loss_accuracy_plot.png')
    if show_plot:
        plt.show()

    df = pd.DataFrame({
        'Epoch': epochs,
        'Validation Loss': loss_list,
        'Validation Accuracy': acc_list
    })
    df.to_csv('csv/val_metrics.csv', index=False)
    print("Validation 결과 저장 완료")
    plt.close()