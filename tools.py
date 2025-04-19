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
    for img_id in img_ids:
        # 이미지 정보 로드 및 전처리
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(sample_image_path, file_name)
        image = Image.open(img_path).convert('RGB')
        image = transform(image)

        # 해당 이미지의 annotation 로드
        ann_ids = coco.getAnnIds(imgIds=img_id)
        if len(ann_ids) == 0:
            continue
        anns = coco.loadAnns(ann_ids)

        # 여러 annotation의 mask를 OR로 통합
        height, width = img_info['height'], img_info['width']
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            mask_ann = coco.annToMask(ann)
            combined_mask = np.maximum(combined_mask, mask_ann)

        # numpy mask -> tensor, (1, H, W)
        mask_tensor = torch.tensor(combined_mask, dtype=torch.float32).unsqueeze(0)
        # 1024x1024에 맞춰 resize
        if height != 1024 or width != 1024:
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=(1024, 1024),
                mode='nearest'
            ).squeeze(0)
        mask_tensor = mask_tensor.squeeze().squeeze()

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


def model_train(model, device, criterion, optimizer, scheduler, dataloader_train, dataloader_valid, epochs=200):
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
            label = label.to(device)  # (B, 1, 1024, 1024)

            optimizer.zero_grad()
            pred = model(img)
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

            pred = model(img)  # (B, 1, 1024, 1024)
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