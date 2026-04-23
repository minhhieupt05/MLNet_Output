import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self, num_classes=5, input_size=64):
        super(MyCNN, self).__init__()
        
        # Bước 1: Định nghĩa các khối trích xuất đặc trưng (Feature Extractor)
        # Phương thức phụ trợ gộp Convolution -> Batch Normalization -> ReLU
        # Quy tắc: Kernel size = 3, Padding = 1 để giữ nguyên kích thước
        def create_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Xếp chồng các khối: 2 khối Conv liên tiếp + 1 Max Pooling để tăng khả năng học
        
        # Cụm 1: Input (3 channels) -> 32 -> 64 -> MaxPool
        self.block1 = nn.Sequential(
            create_conv_block(3, 32),
            create_conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2) # Bước 2: Giảm kích thước đi 1 nửa
        )
        
        # Cụm 2: 64 -> 128 -> 128 -> MaxPool
        self.block2 = nn.Sequential(
            create_conv_block(64, 128),
            create_conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Cụm 3: 128 -> 256 -> 256 -> MaxPool
        self.block3 = nn.Sequential(
            create_conv_block(128, 256),
            create_conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Lặp lại MaxPool cho đến khi Feature Map < 10x10.
        # Ví dụ ảnh đầu vào 64x64, sau 3 lần MaxPool (giảm 2^3 = 8 lần) thì Feature Map là 8x8.
        feature_map_size = input_size // (2 ** 3)
        self.flatten_size = 256 * feature_map_size * feature_map_size
        
        # Bước 3: Chuyển đổi và Phân loại (Classifier)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # Chống Overfitting
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes) 
            # Lưu ý lớp cuối: Số đầu ra = số lượng nhãn và 
            # Không thêm hàm kích hoạt vì sẽ dùng CrossEntropyLoss
        )

    def forward(self, x):
        # Đặc trưng đi qua các khối Feature Extractor
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Dùng hàm view(-1) để làm phẳng (Flatten) Tensor từ 4D (batch, C, H, W) về 2D (batch, FlattenSize)
        x = x.view(x.size(0), -1)
        
        # Đưa qua bộ phân loại
        x = self.classifier(x)
        return x

# Kiểm tra thử model với dummy input
if __name__ == "__main__":
    model = MyCNN(num_classes=5, input_size=64) # Test với ảnh 64x64
    dummy_input = torch.randn(1, 3, 64, 64)   # Batch Size = 1, Channels = 3, W = 64, H = 64
    output = model(dummy_input)
    print("Model Architecture:")
    print(model)
    print(f"\nOutput Shape: {output.shape} (Expected: [1, 5])")
