import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import MyCNN

def prepare_data(data_dir, input_size=64, batch_size=32, validation_split=0.2):
    """
    Bước 4.1: Chuẩn bị DataLoader cho tập Train và Validation
    """
    # Định nghĩa các phép biến đổi (Transforms) cho ảnh
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # Khuyến nghị thêm Normalize nhưng tạm thời để ToTensor cho đơn giản
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Gắn thư mục chứa ảnh (yêu cầu cấu trúc: thư_mục_chính/class_name/ảnh.jpg)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Chia dataset thành Train và Validation
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Khởi tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, dataset.classes

def setup_training_env():
    """Thiết lập môi trường, thiết bị, Loss và Optimizer (Bước 4.2 và Bước 5)"""
    
    # Bước 4.2: Kiểm tra và cấu hình thiết bị tính toán (ưu tiên CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    return device

def main():
    # Thư mục chứa dữ liệu
    data_dir = r"C:\MLNet_Output\Train"
    
    # Chuẩn bị dữ liệu
    print("Preparing data...")
    train_loader, val_loader, classes = prepare_data(data_dir=data_dir, input_size=64)
    print(f"Classes found ({len(classes)}): {classes}")
    
    # Cấu hình thiết bị
    device = setup_training_env()
    
    # Khởi tạo Model
    model = MyCNN(num_classes=len(classes), input_size=64)
    model = model.to(device)
    
    # Bước 5: Chọn hàm mất mát (Criterion) và bộ tối ưu (Optimizer)
    # Dùng CrossEntropyLoss do đây là bài toán phân loại nhiều lớp
    criterion = nn.CrossEntropyLoss()
    
    # Dùng optim.SGD với momentum truyền thống
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    print("Training script setup successfully!")
    print(f"- Loss Function: {criterion.__class__.__name__}")
    print(f"- Optimizer: {optimizer.__class__.__name__} (LR: {learning_rate}, Momentum: 0.9)")
    
    # Bước 6: Chạy vòng lặp huấn luyện (Training Loop)
    num_epochs = 5  # Có thể tùy chỉnh số lượng Epoch
    print(f"\nStarting training for {num_epochs} Epochs...")
    
    for epoch in range(num_epochs):
        model.train() # Đặt mô hình ở chế độ huấn luyện
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        # Duyệt qua từng batch dữ liệu từ DataLoader
        for inputs, labels in train_loader:
            # Đẩy dữ liệu (ảnh và nhãn) lên cùng thiết bị với mô hình
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Xóa gradient cũ
            optimizer.zero_grad()
            
            # Lan truyền tiến: Đưa ảnh qua model để lấy dự đoán
            outputs = model(inputs)
            
            # Tính Loss: So sánh dự đoán với nhãn thực tế
            loss = criterion(outputs, labels)
            
            # Lan truyền ngược: loss.backward() để tính đạo hàm
            loss.backward()
            
            # Cập nhật trọng số
            optimizer.step()
            
            # Thống kê để in ra màn hình
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_preds / total_samples * 100
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}%")
        
    # Bước 1 (Tiếp theo): Lưu mô hình thành file để tái sử dụng
    print("\nSaving model...")
    # Sử dụng pickle như yêu cầu
    with open('image_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully to 'image_model.pkl'")

if __name__ == "__main__":
    main()
