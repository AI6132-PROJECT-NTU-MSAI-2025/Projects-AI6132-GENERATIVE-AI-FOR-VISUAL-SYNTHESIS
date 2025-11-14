import torch
import lpips
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToTensor

# --- 配置 ---
REAL_IMAGES_FOLDER = 'temp_real_images'  # 真实狗狗图的文件夹路径
GENERATED_IMAGES_FOLDER = 'temp_sdxl_generated_images'  # ControlNet生成的狗狗图的文件夹路径
BATCH_SIZE = 8  # 批处理大小，根据显存调整
# -------------
def load_image_tensor(path):
    """加载图像并将其转换为适合 LPIPS 的 PyTorch Tensor"""
    try:
        img = Image.open(path).convert('RGB')
        # LPIPS要求图像张量在 [-1, 1] 范围内
        tensor = ToTensor()(img) * 2 - 1
        return tensor
    except Exception as e:
        print(f"Skipping image {path} due to error: {e}")
        return None

def find_paired_images(folder1, folder2):
    """
    查找两个文件夹中匹配的图像路径。
    文件名相同则配对
    """
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    # 找到共同的文件名
    common_files = sorted(list(files1.intersection(files2)))

    paired_paths = []
    for filename in common_files:
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)
        if os.path.isfile(path1) and os.path.isfile(path2):
            paired_paths.append((path1, path2))

    return paired_paths

def calculate_lpips_score():
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # 加载 LPIPS 模型 (AlexNet 通常是默认且最快的选择)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    print("LPIPS Model (AlexNet) loaded.")
    # 1. 找到配对图像
    paired_paths = find_paired_images(REAL_IMAGES_FOLDER, GENERATED_IMAGES_FOLDER)
    if not paired_paths:
        print("Error: No common, paired images found in both folders.")
        return
    print(f"\nFound {len(paired_paths)} matched image pairs.")
    # 初始化总LPIPS分数列表
    all_lpips_scores = []
    # 2. 分批计算 LPIPS
    num_pairs = len(paired_paths)
    num_batches = int(np.ceil(num_pairs / BATCH_SIZE))
    for i in tqdm(range(num_batches), desc="Calculating LPIPS"):
        batch_paths = paired_paths[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        # 加载并转换图像
        real_tensors = []
        gen_tensors = []
        for real_path, gen_path in batch_paths:
            real_tensor = load_image_tensor(real_path)
            gen_tensor = load_image_tensor(gen_path)
            # 确保两个张量都成功加载
            if real_tensor is not None and gen_tensor is not None:
                # 检查尺寸，LPIPS要求同尺寸
                if real_tensor.shape == gen_tensor.shape:
                    real_tensors.append(real_tensor)
                    gen_tensors.append(gen_tensor)
                else:
                    print(f"Skipping pair {os.path.basename(real_path)}: Image shapes mismatch.")
        if real_tensors:
            # 堆叠张量，并发送到设备
            real_batch = torch.stack(real_tensors).to(device)
            gen_batch = torch.stack(gen_tensors).to(device)
            # 计算 LPIPS 损失/距离
            # loss_fn_alex(tensor_A, tensor_B) 返回一个张量，包含批次中每个元素的LPIPS距离
            lpips_batch = loss_fn_alex(real_batch, gen_batch)
            lpips_list = lpips_batch.squeeze().tolist()

            # 检查 lpips_list 是否为浮点数，如果是，将其包装成列表
            if isinstance(lpips_list, float):
                lpips_list = [lpips_list]

            all_lpips_scores.extend(lpips_list)

    if all_lpips_scores:
        avg_lpips = np.mean(all_lpips_scores)
        # 结果输出
        print("\n=============================================")
        print(f"Average LPIPS Score (Pose/Structure Similarity): {avg_lpips:.4f}")
        print(f"Calculated over {len(all_lpips_scores)} paired images.")
        print(f"Note: Lower is better (0 is identical).")
        print("=============================================")
    else:
        print("No valid image pairs processed.")

if __name__ == '__main__':
    calculate_lpips_score()