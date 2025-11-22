import os
import cv2

def segment_image(img_path):
    """简单占位模型：对图像做灰度化 + 阈值分割"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return mask

def batch_segment(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            continue

        img_path = os.path.join(input_dir, fname)
        save_path = os.path.join(output_dir, fname)

        mask = segment_image(img_path)
        if mask is not None:
            cv2.imwrite(save_path, mask)
            print(f"[OK] {fname} processed.")

if __name__ == "__main__":
    input_dir = "./data/images"
    output_dir = "./results/masks"

    print("Starting batch segmentation...")
    batch_segment(input_dir, output_dir)
    print("Done.")
