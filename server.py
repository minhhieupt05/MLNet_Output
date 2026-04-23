import io
import pickle
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from torchvision import transforms
from PIL import Image

# Để Pickle có thể load lại đối tượng, chúng ta phải có class MyCNN trong scope.
from model import MyCNN 

app = FastAPI(title="Image Classification API")

# 1. Định nghĩa các nhãn lớp (Lấy từ kết quả lúc Train)
class_names = ['Highway', 'Industrial', 'Pasture', 'Residential', 'River']

# 2. Load lại mô hình đã lưu
print("Loading model from image_model.pkl...")
try:
    with open('image_model.pkl', 'rb') as f:
        model = pickle.load(f)
    model.eval()  # Đặt mô hình ở chế độ evaluation
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find image_model.pkl. Have you run the training loop?")
    model = None

# Định nghĩa hàm biến đổi hình ảnh đầu vào y hệt tập huấn luyện
val_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Tạo endpoint @app.get("/") để xuất Giao diện Web
@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    return HTML_CONTENT

# 4. Tạo endpoint @app.post("/predict")
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Nhận dữ liệu hình ảnh được upload, đưa qua mô hình để dự đoán 
    và trả về JSON dưới dạng nhãn lớp kết quả.
    """
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Mô hình chưa được nạp (File chưa tồn tại)."})

    try:
        # Đọc files thành byte
        image_bytes = await file.read()
        
        # Mở hình ảnh bằng công cụ PIL (Pillow)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Tiền xử lý Tensor: Chuyển ảnh qua phép Transform và bọc Batch Size = 1.
        input_tensor = val_transforms(img).unsqueeze(0)
        
        # Đảm bảo tắt Engine gradient bằng cách dùng context torch.no_grad()
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Hàm dự đoán: Max giá trị của Cột (1) là nhãn lớp kết quả
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            
            # Map kết quả lớp trả về tên Lớp.
            predicted_label = class_names[class_idx]
            
        # 4. Trả về kết quả theo chuẩn định dạng JSON
        return {
            "message": f"Kết quả dự đoán: {predicted_label}"
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Lỗi trong quá trình xử lý ảnh: {str(e)}"})

# (Dành cho việc chạy ở Local Server)
if __name__ == "__main__":
    import uvicorn
    # Mở khóa bằng uvicorn server.py:app --reload hoặc chạy python server.py
    uvicorn.run(app, host="127.0.0.0", port=8000)
