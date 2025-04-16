import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import ImprovedCNN
import os
import argparse
from datetime import datetime
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision.datasets import FashionMNIST

transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

train_dataset = FashionMNIST(root='data', train=True, download=True, transform=transform)
test_dataset = FashionMNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FashionMINIST 训练脚本')
    
    # 数据相关参数

    parser.add_argument('--batch_size', type=int, default=64,
                      help='训练批次大小')
    
    # 模型相关参数
    parser.add_argument('--num_classes', type=int, default=10,
                      help='分类数量')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=5,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                      help='训练设备 (cuda/cpu)')
    
    # 优化器相关参数
    parser.add_argument('--optimizer', type=str, default='adam',
                      choices=['adam', 'sgd'],
                      help='优化器选择')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                      help='权重衰减')
    
    # 保存相关参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='模型检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='runs/cifar10_experiment',
                      help='TensorBoard日志目录')
    
    # 加载和保存相关参数
    parser.add_argument('--load_model', type=str, default='checkpoints/best_model.pth',
                      help='加载已有模型的路径')
    parser.add_argument('--save_model', type=str, default='best_model.pth',
                      help='保存模型的固定文件名')
    
    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
    
        # print(images.shape)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    #     print(outputs)
    #     print(predicted[:10])
    #     print(labels[:10])
    #     print(predicted.eq(labels)[:10])
    #     break
    # return 
    epoch_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return epoch_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def main():
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")
    
    # 数据加载
    
    # 创建模型
    model = ImprovedCNN(num_classes=args.num_classes).to(device)
    
    # 如果指定了加载模型，则加载已有模型
    if args.load_model:
        if os.path.exists(args.load_model):
            checkpoint = torch.load(args.load_model, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {args.load_model}")
        else:
            print(f"Warning: Model file {args.load_model} not found. Starting with a new model.")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(args.log_dir)
    
    # 训练循环
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 打印结果
        print(f'Epoch [{epoch+1}/{args.epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            
            # 使用固定文件名
            save_path = os.path.join(args.checkpoint_dir, args.save_model)
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'best_acc': best_acc,
                'args': args
            }, save_path)
            print(f'Model saved to {save_path}')
    
    writer.close()
    print(f"Training completed. Best test accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main() 