import os
import requests
import sys

def predict_image(image_path, server_url="http://localhost:8888/predict"):
    """
    Sử dụng thư viện requests gửi POST request đính kèm dữ liệu (multipart/form-data) 
    tới Server và lấy kết quả trả về.
    """
    # 1. Kiểm tra tồn tại dữ liệu hình ảnh
    if not os.path.exists(image_path):
        print(f"Error: Could not find image file at '{image_path}'")
        return
        
    print(f"Sending image '{os.path.basename(image_path)}' to API Server ({server_url})...")
    
    # 2. Xử lý chuẩn bị dữ liệu hình ảnh Multipart format
    with open(image_path, 'rb') as f:
        # FastAPI endpoint yêu cầu UploadFile ở tham số tên là "file"
        files = {
            "file": (os.path.basename(image_path), f, "image/jpeg")
        }
        
        try:
            # 3. Sử dụng thư viện requests để thực hiện POST request
            response = requests.post(server_url, files=files)
            
            # 4. Nhận phản hồi Response và hiển thị
            if response.status_code == 200:
                result = response.json()
                print("\n[+] Response from Server API:")
                print("========================================")
                print(f" - Message: {result.get('message')}")
            else:
                print(f"[-] API returned error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("\n[-] Connection Error: Could not send request to Server.")
            print("Please ensure you have started uvicorn on Port 8888.")

if __name__ == "__main__":
    # Test tự động: Lấy thông số từ Terminal hoặc chọn 1 ảnh mặc định của tập Train
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Đường dẫn ảnh test ngẫu nhiên (Lấy 1 ảnh nằm trong Dataset của bạn)
        test_image = r"C:\MLNet_Output\Train\Highway\Highway_10.jpg"
        print(f"Using default test image: {test_image}")
        
    predict_image(test_image)
