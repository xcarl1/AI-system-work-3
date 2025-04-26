import os
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# Import the Vision Transformer model
from model import VisionTransformer, resnet
from utils.dataset import PetsDataset

from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard SummaryWriter

# For logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import NNI if available
try:
    import nni
    HAS_NNI = True
except ImportError:
    HAS_NNI = False
    logger.info("NNI not found, hyperparameter tuning will not be available.")

def parse_args():
    parser = argparse.ArgumentParser(description='Vision Transformer Training')
    
    # Model hyperparameters
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for ViT')
    parser.add_argument('--num_classes', type=int, default=37, help='Number of output classes')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # parser.add_argument('--num_blocks', type=list, default=[2, 2, 2, 2], help='[2, 2, 2, 2] or [3, 4, 6, 3]')
    parser.add_argument('--initial_filter_size', type=int, default=64, help='64 or 128')
    parser.add_argument('--model_type', type=str, default='resnet18')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', 
                        choices=['step', 'cosine', 'plateau'], help='Learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=30, help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--exp_name', type=str, default="nni")

    # Dataset and data loading
    parser.add_argument('--data_dir', type=str, default='/root/siton-data-412581749c3f4cfea0d7c972b8742057/proj/work_3_xzp/data', help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    # Optimization and regularization
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], 
                        help='Optimizer to use')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--mixup', action='store_true', help='Use mixup data augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Alpha parameter for mixup')

    # Device and hardware settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use automatic mixed precision')

    # Checkpointing and logging
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--runs', type=str, default='./runs', help='Directory to tensorboard')
    parser.add_argument('--log_interval', type=int, default=10, help='Print interval')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint frequency (epochs)')

    # NNI hyperparameter tuning
    parser.add_argument('--use_nni', action='store_true', default=True, help='Use NNI for hyperparameter tuning')

    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    return args

def get_dataset(args):
    """
    Load the appropriate dataset based on args.dataset
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),    # 调整图像大小
        transforms.ToTensor(),            # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    
    
    train_dataset = PetsDataset(args.data_dir, "train", transform=transform)
    test_dataset = PetsDataset(args.data_dir, "test", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, test_loader

def mixup_data(x, y, alpha=1.0):
    """
    Applies mixup augmentation to the batch
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_optimizer(model, args):
    """
    Create optimizer based on args
    """
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                             momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    return optimizer

def get_scheduler(optimizer, args):
    """
    Create learning rate scheduler based on args
    """
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_gamma, 
                                                     patience=10, verbose=True)
    else:
        raise ValueError(f"Unsupported scheduler: {args.lr_scheduler}")
    
    return scheduler

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, args):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply mixup if enabled
        if args.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup_alpha)
            
        optimizer.zero_grad()
        
        # Use AMP if enabled
        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if args.mixup:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if args.mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        if not args.mixup:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % args.log_interval == 0:
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total if not args.mixup else 'N/A',
                'lr': optimizer.param_groups[0]['lr']
            })
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total if not args.mixup else None
    
    return train_loss, train_acc

def validate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def save_checkpoint(state, filename):
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")

def main():
    args = parse_args()
    
    # If using NNI, update arguments with trial parameters
    if args.use_nni and HAS_NNI:
        params = nni.get_next_parameter()
        args.__dict__.update(params)
        logger.info(f"NNI trial parameters: {params}")
    elif args.use_nni and not HAS_NNI:
        logger.warning("NNI not found but --use_nni flag is set. Proceeding without NNI.")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load datasets
    train_loader, test_loader = get_dataset(args)
    # logger.info(f"Dataset: {args.dataset} with {args.num_classes} classes")
    
    # Create model
    if args.model_type == 'resnet18':
        model = resnet(num_classes=args.num_classes, num_blocks=[2, 2, 2, 2])
    elif args.model_type == 'resnet34':
        model = resnet(num_classes=args.num_classes, num_blocks=[3, 4, 6, 3])
    else:
        model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Get optimizer and scheduler
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint.get('best_acc', 0)
            logger.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            logger.error(f"No checkpoint found at '{args.resume}'")
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Training configuration: {args}")
    # Set up TensorBoard writer
    log_dir = os.path.join(args.runs, args.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)  # 设置 TensorBoard 保存路径
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args)
        
        # Evaluate on test set
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%" if train_acc is not None else f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Update scheduler
        if args.lr_scheduler == 'plateau':
            scheduler.step(test_loss)
        else:
            scheduler.step()
        
        # Save model if it's the best so far
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_acc': best_acc,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'PetClasses_best.pth'))
        
        # Periodically save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_acc': best_acc,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'PetClasses_epoch{epoch+1}.pth'))
        
        # Report intermediate result to NNI if enabled
        if args.use_nni and HAS_NNI:
            nni.report_intermediate_result(test_acc)
    
    # Report final result to NNI if enabled
    if args.use_nni and HAS_NNI:
        nni.report_final_result(best_acc)
    
    logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%")
    
    # Save final model
    save_checkpoint({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'args': vars(args)
    }, os.path.join(args.save_dir, f'PetClasses_final.pth'))

# 在文件末尾修改代码，添加运行时间记录
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"总运行时间: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"训练完成! 总运行时间: {int(hours):02d}小时 {int(minutes):02d}分钟 {seconds:.2f}秒")