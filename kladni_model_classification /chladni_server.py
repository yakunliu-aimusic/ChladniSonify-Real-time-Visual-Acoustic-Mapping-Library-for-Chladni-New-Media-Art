import socket       # Network communication (Socket)
import threading    # Multi-threaded concurrent request handling
import io           # Byte stream handling (convert received binary data to image)
import json         # Serialize recognition results as JSON response
from PIL import Image  # Image processing (open and convert images)
import torch        # PyTorch framework for loading and running the AI model
from torchvision import transforms  # Image preprocessing utilities

# === Configuration ===
IMAGE_PORT = 9999      # Port for receiving images
RESPONSE_PORT = 9998   # Port for sending frequency results
MODEL_PATH = "cbam_chladni_model.pth"  # Path to pretrained model weights
FREQ_TABLE = [
    163, 257, 305, 323, 412, 489, 521, 603,
    678, 734, 812, 905, 1023, 1156, 1342
]  # Frequencies for 15 classes

# === Image preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to 224x224 (model input size)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Load model ===
device = torch.device("cpu")
model = CBAMResNet18(num_classes=15)  # Ensure this model class is imported
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def handle_image(conn_img, conn_resp):
    try:
        # Receive complete JPEG (terminated by 0xFFD9)
        data = b""
        while True:
            chunk = conn_img.recv(4096)
            if not chunk:
                break
            data += chunk
            if data.endswith(b'\xff\xd9'):
                break

        # AI inference
        x = preprocess(data)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1).item()
            freq = FREQ_TABLE[pred]

        # Send JSON response
        response = json.dumps({"label": pred, "freq": freq}) + "\n"
        conn_resp.sendall(response.encode())

    except Exception as e:
        print(f"[ERROR] {e}")
        conn_resp.sendall(json.dumps({"error": str(e)}) + "\n")
    finally:
        conn_img.close()
        conn_resp.close()

def start_servers():
    # Image reception server
    sock_img = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_img.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_img.bind(("127.0.0.1", IMAGE_PORT))
    sock_img.listen(1)

    # Response server
    sock_resp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_resp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_resp.bind(("127.0.0.1", RESPONSE_PORT))
    sock_resp.listen(1)

    print(f"OK Waiting for Max on ports {IMAGE_PORT} (image) and {RESPONSE_PORT} (response)...")

    while True:
        conn_img, _ = sock_img.accept()
        conn_resp, _ = sock_resp.accept()
        thread = threading.Thread(target=handle_image, args=(conn_img, conn_resp))
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    start_servers()