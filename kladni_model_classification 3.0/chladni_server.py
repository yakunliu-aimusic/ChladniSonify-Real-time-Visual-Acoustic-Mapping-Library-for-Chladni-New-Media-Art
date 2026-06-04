import socket       # 实现网络通信（Socket）
import threading    # 多线程处理，支持并发请求
import io           # 字节流处理（把接收的二进制数据转成图像）
import json         # 把识别结果转成JSON格式回传
from PIL import Image  # 图像处理（打开、转换图像）
import torch        # PyTorch框架，加载/运行AI模型
from torchvision import transforms  # 图像预处理工具

# === 配置 ===
IMAGE_PORT = 9999      # 接收图像的端口
RESPONSE_PORT = 9998   # 发送频率结果的端口
MODEL_PATH = "cbam_chladni_model.pth"  # 预训练AI模型文件路径
FREQ_TABLE = [
    163, 257, 305, 323, 412, 489, 521, 603,
    678, 734, 812, 905, 1023, 1156, 1342
]  # 15类对应频率

# === 图像预处理 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # 缩放到224×224（模型输入尺寸）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === 加载模型 ===
device = torch.device("cpu")
model = CBAMResNet18(num_classes=15)  # 请确保此模型定义已导入
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def handle_image(conn_img, conn_resp):
    try:
        # 接收完整 JPEG（以 0xFFD9 结尾）
        data = b""
        while True:
            chunk = conn_img.recv(4096)
            if not chunk:
                break
            data += chunk
            if data.endswith(b'\xff\xd9'):
                break

        # AI 推理
        x = preprocess(data)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1).item()
            freq = FREQ_TABLE[pred]

        # 回传 JSON
        response = json.dumps({"label": pred, "freq": freq}) + "\n"
        conn_resp.sendall(response.encode())

    except Exception as e:
        print(f"[ERROR] {e}")
        conn_resp.sendall(json.dumps({"error": str(e)}) + "\n")
    finally:
        conn_img.close()
        conn_resp.close()

def start_servers():
    # 图像接收服务器
    sock_img = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_img.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_img.bind(("127.0.0.1", IMAGE_PORT))
    sock_img.listen(1)

    # 响应发送服务器
    sock_resp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_resp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_resp.bind(("127.0.0.1", RESPONSE_PORT))
    sock_resp.listen(1)

    print(f"✅ Waiting for Max on ports {IMAGE_PORT} (image) and {RESPONSE_PORT} (response)...")

    while True:
        conn_img, _ = sock_img.accept()
        conn_resp, _ = sock_resp.accept()
        thread = threading.Thread(target=handle_image, args=(conn_img, conn_resp))
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    start_servers()