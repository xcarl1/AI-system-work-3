import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class PetsDataset(Dataset):
    def __init__(self, base_data_dir, split, transform=None):
        self.base_data_dir = base_data_dir
        self.transform = transform
        images = []
        labels = []
        print("loading " + split + " data...")
        if split == "train":
            train_txt_data = os.path.join(self.base_data_dir, "annotations/trainval.txt")
            with open(train_txt_data, "r") as file:
                 for line in file:
                     parts = line.strip().split()
                     image_name = parts[0] + ".jpg"
                     label = int(parts[1])
                     image_path = os.path.join(self.base_data_dir, "images/" + image_name)
                     images.append(image_path)
                     labels.append(label)
        if split == "test":
            train_txt_data = os.path.join(self.base_data_dir, "annotations/test.txt")
            with open(train_txt_data, "r") as file:
                 for line in file:
                     parts = line.strip().split()
                     image_name = parts[0] + ".jpg"
                     label = int(parts[1])
                     image_path = os.path.join(self.base_data_dir, "images/" + image_name)
                     images.append(image_path)
                     labels.append(label)
        
        self.images = images
        self.labels = labels
        print(str(len(self.images)) + " images are loaded in total")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, label - 1



if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),    # 调整图像大小
        transforms.ToTensor(),            # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    
    base_data_dir = "/root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_3_xzp/data"
    
    dataset = PetsDataset(base_data_dir, "test", transform = transform)
    
    print(dataset[0][0].shape)
    print(dataset[0][1])
    
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    