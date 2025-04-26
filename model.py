import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 ViT 模型
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=10, embed_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super(VisionTransformer, self).__init__()

        # 计算图像的 patch 数量
        self.num_patches = (image_size // patch_size) ** 2

        # Patch Embedding
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional Encoding
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # +1 for the class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Class token
        
        # Transformer Encoder Layers
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim*4, 
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Classifier head (final linear layer)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Step 1: Patch Embedding
        x = self.patch_embed(x)  # (batch_size, embed_dim, patches, patches)
        x = x.flatten(2)  # Flatten the patches (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)

        # Step 2: Add class token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches+1, embed_dim)

        # Step 3: Add positional encoding
        x = x + self.position_embeddings
        
        # Step 4: Pass through Transformer Encoder layers
        for transformer in self.transformer_blocks:
            x = transformer(x)

        # Step 5: Classification head
        cls_output = x[:, 0]  # Use the output corresponding to the class token
        out = self.fc(cls_output)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, input_channels=3, initial_filter_size=64):
        super(ResNet, self).__init__()
        self.in_channels = initial_filter_size
        
        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual Blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet(num_classes=1000, num_blocks=[2, 2, 2, 2], input_channels=3, initial_filter_size=64):
    return ResNet(BasicBlock, num_blocks, num_classes, input_channels, initial_filter_size)



if __name__ == '__main__':
    # 示例：创建一个 ViT 模型并查看其结构
    model = VisionTransformer(image_size=224, patch_size=16, num_classes=10, embed_dim=256, num_heads=8, num_layers=6, dropout=0.1)
    print(model)

    # 输入一个示例图像，假设图像的大小是 (batch_size, 3, 224, 224)
    sample_input = torch.randn(8, 3, 224, 224)  # batch_size=8
    output = model(sample_input)
    print(output.shape)  # 输出的形状应该是 (batch_size, num_classes)
