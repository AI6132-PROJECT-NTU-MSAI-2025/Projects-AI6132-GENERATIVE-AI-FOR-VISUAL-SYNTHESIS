## 请在运行了inference_FID_KID_CLIP.py 并完成图片生成后运行本代码
import torch
import os
import shutil
import numpy as np
import datetime
from PIL import Image
from diffusers import StableDiffusionXLPipeline, UniPCMultistepScheduler

# 评估指标库
from torchmetrics.functional.multimodal import clip_score
import cleanfid.fid as fid


# 注意：LPIPS 在此 SDXL 基线评估中不适用（它被设计来评估结构保真度），因此我们不导入 lpips

# --- Pipeline 初始化：纯 SDXL 基线 ---
def initialize_sdxl_pipeline(base_model_path, DEVICE):
    """初始化纯 SDXL Pipeline"""
    print("--- 正在加载 纯 SDXL 评估 Pipeline ---")
    pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
        base_model_path, torch_dtype=torch.float16, variant="fp16"
    )
    pipe_sdxl.scheduler = UniPCMultistepScheduler.from_config(pipe_sdxl.scheduler.config)
    pipe_sdxl.to(DEVICE)
    pipe_sdxl.enable_xformers_memory_efficient_attention()
    return pipe_sdxl


# =======================================================
# 阶段一：纯 SDXL 基线图像生成 (耗时操作)
# =======================================================
def sdxl_baseline_generation_phase(pipe_sdxl, TEMP_SDXL_GEN_DIR, TEMP_PROMPTS_FILE, IMAGE_SIZE,
                                   num_inference_steps, NUM_SAMPLES_TO_EVALUATE, DEVICE):
    """纯 SDXL 基线生成循环：从 Prompts 文件中读取，生成图像并保存"""

    # 清理和创建临时目录
    if os.path.exists(TEMP_SDXL_GEN_DIR): shutil.rmtree(TEMP_SDXL_GEN_DIR)
    os.makedirs(TEMP_SDXL_GEN_DIR, exist_ok=True)

    # 确保 Prompts 文件存在并加载
    if not os.path.exists(TEMP_PROMPTS_FILE):
        raise FileNotFoundError(
            f"Prompts file not found at {TEMP_PROMPTS_FILE}. 请确保 ControlNet 脚本已运行并生成此文件。")
    with open(TEMP_PROMPTS_FILE, 'r', encoding='utf-8') as f:
        clip_prompts = [line.strip() for line in f.readlines()]

    num_to_generate = min(len(clip_prompts), NUM_SAMPLES_TO_EVALUATE)
    print(f"\n--- 阶段一：开始生成 {num_to_generate} 张纯 SDXL 基线图片 ---")

    for i in range(num_to_generate):
        prompt = clip_prompts[i]

        # 1. SDXL 图像生成
        with torch.no_grad():
            generated_image = pipe_sdxl(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(DEVICE).manual_seed(i + 1)
            ).images[0]

        # 2. 保存图片到临时文件夹
        generated_image.resize(IMAGE_SIZE).save(os.path.join(TEMP_SDXL_GEN_DIR, f"{i:05d}.png"))

        if (i + 1) % 100 == 0:
            print(f"SDXL 已处理 {i + 1} 个样本...")

    print("--- 纯 SDXL 图像生成完成 ---")


# =======================================================
# 阶段二：指标计算 (快速操作)
# =======================================================
def metric_phase(TEMP_GEN_DIR, TEMP_REAL_DIR, TEMP_PROMPTS_FILE, DEVICE):
    """计算 FID, KID 和 CLIP Score"""

    # 1. 加载 Prompts
    if not os.path.exists(TEMP_PROMPTS_FILE):
        raise FileNotFoundError(f"Prompts file not found at {TEMP_PROMPTS_FILE}.")
    with open(TEMP_PROMPTS_FILE, 'r', encoding='utf-8') as f:
        clip_prompts = [line.strip() for line in f.readlines()]

    num_samples = len(clip_prompts)
    print(f"\n--- 开始计算指标，共 {num_samples} 个样本 ---")

    # --- 1. 检查文件 ---
    if not os.path.exists(TEMP_REAL_DIR) or len(os.listdir(TEMP_REAL_DIR)) != num_samples:
        print(f"错误：缺少 Real Images 文件夹 ({TEMP_REAL_DIR}) 或样本数量不匹配。跳过 FID/KID。")
        fid_value = 0.0
        kid_value = 0.0
        run_fid_kid = False
    else:
        run_fid_kid = True

    # --- 2. 计算 FID/KID ---
    print("\n--- 2. 计算 FID 和 KID ---")

    if run_fid_kid and num_samples >= 50:
        metric_params = dict(fdir1=TEMP_REAL_DIR, fdir2=TEMP_GEN_DIR, device=DEVICE,
                             num_workers=8, batch_size=64, mode="clean")

        fid_value = fid.compute_fid(**metric_params)
        kid_results = fid.compute_kid(**metric_params)
        kid_value = kid_results[0] if isinstance(kid_results, (tuple, list)) else kid_results
        print(f"  > FID: {fid_value:.4f}, KID: {kid_value:.4f}")
    else:
        if run_fid_kid:
            print(f"  > 样本数量 ({num_samples}) 过少，跳过 FID/KID 计算。")
        fid_value = 0.0
        kid_value = 0.0

    # --- 3. 计算 CLIP Score (分批计算) ---
    print("\n--- 3. 计算 CLIP Score ---")
    generated_files = [os.path.join(TEMP_GEN_DIR, f) for f in os.listdir(TEMP_GEN_DIR) if f.endswith('.png')]
    generated_files.sort()

    CLIP_BATCH_SIZE = 64
    total_clip_score_sum = 0.0

    for start_idx in range(0, num_samples, CLIP_BATCH_SIZE):
        end_idx = min(start_idx + CLIP_BATCH_SIZE, num_samples)
        current_batch_size = end_idx - start_idx

        current_files = generated_files[start_idx:end_idx]
        current_prompts = clip_prompts[start_idx:end_idx]

        current_tensors = []
        for f in current_files:
            img = Image.open(f).convert("RGB").resize((224, 224))
            tensor = torch.tensor(np.array(img), dtype=torch.uint8).permute(2, 0, 1)  # (C, H, W)
            current_tensors.append(tensor)

        images_tensor = torch.stack(current_tensors).to(DEVICE)

        with torch.no_grad():
            clip_score_val_batch = clip_score(images_tensor, current_prompts).item()
            total_clip_score_sum += clip_score_val_batch * current_batch_size

        del images_tensor
        torch.cuda.empty_cache()

    final_clip_score = (total_clip_score_sum / num_samples) / 100  # 转换为 [0, 1] 范围
    print(f"  > CLIP Score 计算完成。")

    return final_clip_score, fid_value, kid_value


if __name__ == "__main__":

    # --- 【配置区】请根据您的环境和需求修改 ---
    NUM_SAMPLES_TO_EVALUATE = 1200
    base_model_path = "./SDXL_base_fp16"  # 您的 SDXL 模型路径
    num_inference_steps = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_SIZE = (512, 512)

    # 临时目录和文件 (依赖于您原 ControlNet 脚本的输出)
    TEMP_SDXL_GEN_DIR = "temp_sdxl_generated_images"  # 新生成的 SDXL 图像目录
    TEMP_REAL_DIR = "temp_real_images"  # 真实图像目录 (由原脚本生成)
    TEMP_PROMPTS_FILE = "temp_clip_prompts.txt"  # Prompt 文件 (由原脚本生成)

    OUTPUT_FILE_NAME = "sdxl_baseline_evaluation_results.txt"
    # --- 【配置区结束】---

    # --- 检查依赖文件 ---
    if not os.path.exists(TEMP_PROMPTS_FILE):
        raise FileNotFoundError(
            f"致命错误：未找到 Prompts 文件 {TEMP_PROMPTS_FILE}。请先运行原 ControlNet 脚本的生成阶段。")

    if not os.path.exists(TEMP_REAL_DIR):
        print(f"警告：未找到真实图像目录 {TEMP_REAL_DIR}。FID/KID 将被跳过，但 CLIP Score 仍可计算。")

    # --- 0. 模型初始化 ---
    pipe_sdxl_baseline = initialize_sdxl_pipeline(base_model_path, DEVICE)

    # =======================================================
    # I. 图像生成
    # =======================================================
    print("\n" + "=" * 50)
    print("             阶段一：纯 SDXL 基线图像生成 ")
    print("=" * 50)
    sdxl_baseline_generation_phase(
        pipe_sdxl_baseline, TEMP_SDXL_GEN_DIR, TEMP_PROMPTS_FILE, IMAGE_SIZE,
        num_inference_steps, NUM_SAMPLES_TO_EVALUATE, DEVICE
    )

    # 释放 VRAM
    del pipe_sdxl_baseline
    torch.cuda.empty_cache()

    # =======================================================
    # II. 指标计算
    # =======================================================
    print("\n" + "=" * 50)
    print("              阶段二：纯 SDXL 基线指标计算 ")
    print("=" * 50)

    sdxl_clip_score, sdxl_fid, sdxl_kid = metric_phase(
        TEMP_SDXL_GEN_DIR, TEMP_REAL_DIR, TEMP_PROMPTS_FILE, DEVICE
    )

    # --- 最终打印和保存结果 ---
    print("\n==================================")
    print("      纯 SDXL 基线评估结果        ")
    print("==================================")
    print(f"CLIP Score (文本一致性, 越高越好): {sdxl_clip_score:.4f}")
    print(f"FID Score (图像质量/多样性, 越低越好): {sdxl_fid:.4f}")
    print(f"KID Score (图像质量/多样性, 越低越好): {sdxl_kid:.4f}")
    print("==================================")

    with open(OUTPUT_FILE_NAME, 'w') as f:
        f.write("--- 纯 SDXL 基线评估结果 ---\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Model Path: {base_model_path}\n")
        f.write(f"Evaluation Samples: {NUM_SAMPLES_TO_EVALUATE}\n")
        f.write("-" * 30 + "\n")
        f.write(f"CLIP Score (越高越好): {sdxl_clip_score:.4f}\n")
        f.write(f"FID Score (越低越好): {sdxl_fid:.4f}\n")
        f.write(f"KID Score (越低越好): {sdxl_kid:.4f}\n")
        f.write("-" * 30 + "\n")

    print(f"\n 纯 SDXL 评估结果已成功保存到文件: {OUTPUT_FILE_NAME}")