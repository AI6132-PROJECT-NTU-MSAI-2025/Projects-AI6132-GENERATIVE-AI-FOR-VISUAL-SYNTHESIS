import torch
import os
import shutil
import io
import numpy as np
import datetime
from PIL import Image
from datasets import load_dataset


from Adapter.Sampling import diffusion_inference
from Adapter.inference_base import get_base_argument_parser
from Adapter.extra_condition.api import get_cond_animalpose

from omegaconf import OmegaConf
import cv2
from configs.utils import instantiate_from_config
from basicsr.utils import img2tensor
import copy

# 评估指标库
from torchmetrics.functional.multimodal import clip_score
import cleanfid.fid as fid 
import lpips

parser = get_base_argument_parser()
global_opt = parser.parse_args()
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def extract_and_decode_image(image_data):
    """尝试从字节流或包含字节流的字典中提取数据，并转换为 PIL.Image 对象。"""
    image_bytes = None
    if isinstance(image_data, dict):
        if 'bytes' in image_data:
            image_bytes = image_data['bytes']
        elif 'data' in image_data:
            image_bytes = image_data['data']
        else:
            raise ValueError(f"Image dict found but missing expected keys ('bytes' or 'data'): {image_data.keys()}")
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    else:
        # 假设它已经是 PIL Image
        return image_data 
    
    # 使用 Image.open(io.BytesIO(...)) 解码图像
    return Image.open(io.BytesIO(image_bytes))

# --- 辅助函数：PIL 转 LPIPS Tensor ---
def pil_to_lpips_tensor(img: Image.Image, device) -> torch.Tensor:
    """将 PIL Image 转换为 LPIPS 期望的 [-1, 1] 范围的 Tensor (C, H, W)"""
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 # [0, 1]
    tensor = tensor * 2.0 - 1.0 # 转换到 [-1, 1]
    return tensor.unsqueeze(0).to(device)


# --- Pipeline 初始化 ---
def initialize_basemodel_and_adapter(base_model_path, adapter_config_name, DEVICE):
    """初始化 ControlNet Pipeline"""
    print(f"start loading basemodel from:{base_model_path}")
    sampler = diffusion_inference(base_model_path)
    print(f"basemodel {base_model_path} is loaded successfully")
    print(f"start loading config file:configs/inference/Adapter-XL-{adapter_config_name}.yaml")
    config = OmegaConf.load(f'configs/inference/Adapter-XL-{adapter_config_name}.yaml')
    adapter_config = config.model.params.adapter_config
    adapter = instantiate_from_config(adapter_config).to(DEVICE)
    adapter.load_state_dict(torch.load(config.model.params.adapter_config.pretrained))
    print("adapter is successfully loaded")

    return sampler, adapter


def generate_image(sampler, adapter, prompt, a_prompt, n_prompt, ddim_steps, con_strength, cond_image=None, seed=-1):
    prompt = prompt + ', ' + a_prompt
    cond_tensor = get_cond_animalpose(global_opt, cond_image)

    adapter_features = adapter(cond_tensor)
    for i in range(len(adapter_features)):
        adapter_features[i] = adapter_features[i] * con_strength

    result_np_bgr = sampler.inference(
        prompt=prompt,
        size=(cond_image.shape[-2], cond_image.shape[-1]),
        prompt_n=n_prompt,
        steps=ddim_steps,
        adapter_features=copy.deepcopy(adapter_features),
        seed=seed,
    )

    result_np_rgb = result_np_bgr[:, :, ::-1]  # BGR -> RGB
    result_pil = Image.fromarray(result_np_rgb.astype('uint8'))  # NumPy -> PIL

    return result_pil

# =======================================================
# 阶段一：图像生成和 LPIPS 计算 (耗时操作)
# =======================================================
def generation_phase(sampler, adapter, dataset, TEMP_GEN_DIR, TEMP_REAL_DIR, TEMP_PROMPTS_FILE, TEMP_LPIPS_FILE, IMAGE_SIZE,
                     caption_col, conditioning_image_col, real_image_col, 
                     control_strength, num_inference_steps, NUM_SAMPLES_TO_EVALUATE, 
                    DEVICE, lpips_model):
    """主生成循环：生成图像，保存，计算结构 LPIPS，并保存 Prompts 和 LPIPS 分数"""
    
    # 清理和创建临时目录
    if os.path.exists(TEMP_GEN_DIR): shutil.rmtree(TEMP_GEN_DIR)
    if os.path.exists(TEMP_REAL_DIR): shutil.rmtree(TEMP_REAL_DIR)
    os.makedirs(TEMP_GEN_DIR, exist_ok=True)
    os.makedirs(TEMP_REAL_DIR, exist_ok=True)

    clip_prompts = []
    structural_lpips_scores = [] 
    
    data_iterator = dataset.take(NUM_SAMPLES_TO_EVALUATE)
    print(f"\n--- 阶段一：开始从数据集中生成和收集 {NUM_SAMPLES_TO_EVALUATE} 张图片 ---")

    for i, example in enumerate(data_iterator):
        prompt = example[caption_col]
        raw_control_img = extract_and_decode_image(example[conditioning_image_col])
        raw_real_img = extract_and_decode_image(example[real_image_col])
        
        control_img = raw_control_img.convert("RGB").resize(IMAGE_SIZE)
        real_img = raw_real_img.convert("RGB").resize(IMAGE_SIZE)

        control_img_np = np.array(control_img)
        # 1. ControlNet 图像生成
        with torch.no_grad():
            a_prompt = "in real world, high quality"
            n_prompt = "extra digit, fewer digits, cropped, worst quality, low quality"
            generated_image = generate_image(prompt=prompt,
                                             sampler=sampler,
                                             adapter=adapter,
                                             a_prompt=a_prompt,
                                             n_prompt=n_prompt,
                                             ddim_steps=num_inference_steps,
                                             con_strength=control_strength,
                                             seed=-1,
                                             cond_image=control_img_np
                                             )

        # 2. 结构 LPIPS 计算
        generated_image = generated_image.resize(IMAGE_SIZE)

        generated_image_tensor = pil_to_lpips_tensor(generated_image, DEVICE)
        real_img_tensor = pil_to_lpips_tensor(real_img, DEVICE)

        with torch.no_grad():
            lpips_score = lpips_model(generated_image_tensor, real_img_tensor).item()
        
        structural_lpips_scores.append(lpips_score)

        # 3. 保存图片到临时文件夹 (文件名使用索引，保持一致)
        generated_image.save(os.path.join(TEMP_GEN_DIR, f"{i:05d}.png"))
        real_img.save(os.path.join(TEMP_REAL_DIR, f"{i:05d}.png"))
        
        # 4. 收集 prompts
        clip_prompts.append(prompt)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 个样本...")
            
    avg_structural_lpips = sum(structural_lpips_scores) / len(structural_lpips_scores) if structural_lpips_scores else 0
    print("--- 图像生成和 LPIPS 结构保真度计算完成 ---")

    # 5. 保存 Prompts 
    with open(TEMP_PROMPTS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(clip_prompts))
    print(f"--- Prompts 已保存到 {TEMP_PROMPTS_FILE} ---")
    
    # 6. 保存 LPIPS 分数
    with open(TEMP_LPIPS_FILE, 'w', encoding='utf-8') as f:
        f.write(f"AVG_LPIPS: {avg_structural_lpips}\n")
        f.write("-" * 30 + "\n")
        f.write('\n'.join([f"{score:.6f}" for score in structural_lpips_scores]))
    print(f"--- 结构 LPIPS 分数已保存到 {TEMP_LPIPS_FILE} ---")
    
    return avg_structural_lpips

# =======================================================
# 阶段二：指标计算 (快速操作) - 针对 CLIP Score 进行了分批优化
# =======================================================
def metric_phase(TEMP_GEN_DIR, TEMP_REAL_DIR, TEMP_PROMPTS_FILE, TEMP_LPIPS_FILE, DEVICE, avg_structural_lpips_from_phase1=None):
    """计算 FID, KID 和 CLIP Score"""
    
    # 0. 加载或使用 LPIPS 分数
    final_lpips_score = avg_structural_lpips_from_phase1
    if final_lpips_score is None:
        if os.path.exists(TEMP_LPIPS_FILE):
            print(f"--- 从文件 {TEMP_LPIPS_FILE} 加载 LPIPS 分数 ---")
            with open(TEMP_LPIPS_FILE, 'r') as f:
                line = f.readline().strip()
                if line.startswith("AVG_LPIPS: "):
                    final_lpips_score = float(line.split("AVG_LPIPS: ")[1])
                else:
                    raise ValueError(f"LPIPS file {TEMP_LPIPS_FILE} format error.")
        else:
             raise FileNotFoundError(f"LPIPS file not found at {TEMP_LPIPS_FILE}. Please run generation phase or pass LPIPS manually.")
    
    # 1. 加载 Prompts
    if not os.path.exists(TEMP_PROMPTS_FILE):
        raise FileNotFoundError(f"Prompts file not found at {TEMP_PROMPTS_FILE}. Run generation phase first.")
    with open(TEMP_PROMPTS_FILE, 'r', encoding='utf-8') as f:
        clip_prompts = [line.strip() for line in f.readlines()]
    
    num_samples = len(clip_prompts)
    print(f"\n--- 阶段二：计算指标，共 {num_samples} 个样本 ---")

    # --- 1. 计算 FID/KID ---
    print("\n--- 1. 计算 FID 和 KID ---")
    
    # 检查样本数量是否足够（至少 50 或 100，否则可能报错或结果不准确）
    if num_samples < 50: 
        print(f"样本数量 ({num_samples}) 过少，跳过 FID/KID 计算")
        fid_value = 0.0
        kid_value = 0.0
    else:
        metric_params = dict(fdir1=TEMP_REAL_DIR, fdir2=TEMP_GEN_DIR, device=DEVICE,
                             num_workers=8, batch_size=64, mode="clean")

        fid_value = fid.compute_fid(**metric_params)
        kid_results = fid.compute_kid(**metric_params)
        # 提取 KID 的均值
        kid_value = kid_results[0] if isinstance(kid_results, (tuple, list)) else kid_results


    # --- 2. 计算 CLIP Score (优化后的分批计算) ---
    print("\n--- 2. 计算 CLIP Score (分批处理以避免 OOM) ---")
    generated_files = [os.path.join(TEMP_GEN_DIR, f) for f in os.listdir(TEMP_GEN_DIR) if f.endswith('.png')]
    generated_files.sort()
    
    if len(generated_files) != num_samples:
        print(f"⚠️ 警告：实际找到的图片数量 ({len(generated_files)}) 与 Prompt 数量 ({num_samples}) 不一致。")
    
    # 设置 Batch Size，根据您的 VRAM 剩余量调整，通常 64 或 128 较为安全
    CLIP_BATCH_SIZE = 64  
    total_clip_score_sum = 0.0
    
    # 手动分批迭代
    for start_idx in range(0, num_samples, CLIP_BATCH_SIZE):
        end_idx = min(start_idx + CLIP_BATCH_SIZE, num_samples)
        current_batch_size = end_idx - start_idx
        
        # 提取当前批次的图片和 Prompts
        current_files = generated_files[start_idx:end_idx]
        current_prompts = clip_prompts[start_idx:end_idx]
        
        current_tensors = []
        for f in current_files:
            img = Image.open(f).convert("RGB").resize((224, 224))
            # PIL Image 转 torch.uint8 Tensor
            tensor = torch.tensor(np.array(img), dtype=torch.uint8).permute(2, 0, 1) # (C, H, W)
            current_tensors.append(tensor)
            
        images_tensor = torch.stack(current_tensors).to(DEVICE)
        
        # 在 no_grad 环境下计算 CLIP Score
        with torch.no_grad():
            # clip_score 函数返回百分比 [0, 100] 范围
            clip_score_val_batch = clip_score(images_tensor, current_prompts).item() 
            # 累加分数总和（非平均值）
            total_clip_score_sum += clip_score_val_batch * current_batch_size

        # 释放当前批次的 GPU 内存
        del images_tensor
        torch.cuda.empty_cache()
        
        if (end_idx) % 500 == 0 or end_idx == num_samples:
            print(f"  > 已计算 {end_idx} / {num_samples} 个样本的 CLIP Score...")

    # 计算最终的平均 CLIP Score
    final_clip_score = (total_clip_score_sum / num_samples) / 100 # 除以 100 转换为 [0, 1] 范围
    print(f"  > CLIP Score 分批计算完成。")


    return final_clip_score, fid_value, kid_value, final_lpips_score


if __name__ == "__main__":

    # 设置为 True 运行生成阶段，False 跳过生成，直接从磁盘加载已经生成完成的图片集
    RUN_GENERATION_PHASE = False  
    
    # 如果此前已完成图像生成步骤，这里设置为None，会直接从lpip记录文件读取数值
    avg_structural_lpips_result = None # 初始设置为 None

    # --- 全局配置 ---
    PARQUET_FILE_PATH = "combined_test_data.parquet"
    conditioning_image_col = 'conditioning_image'
    real_image_col = 'original_image'
    caption_col = "caption"
    
    # 评估样本数量：请根据测试需求修改。
    # 快速测试可设置为 50；正式评估 FID/KID 至少应为 1000。
    NUM_SAMPLES_TO_EVALUATE = 1200 
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEMP_GEN_DIR = "temp_t2iadapter_generated_images"
    TEMP_REAL_DIR = "temp_real_images"
    TEMP_PROMPTS_FILE = "temp_clip_prompts.txt"
    TEMP_LPIPS_FILE = "temp_structural_lpips_scores.txt" # LPIPS 结果保存文件
    
    # 推荐使用 SDXL 尺寸
    IMAGE_SIZE = (512, 512) 

    base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    adapter_config_name = "animalpose"
    control_strength = 0.7
    num_inference_steps = 20
    OUTPUT_FILE_NAME = "t2iadapter_evaluation_results.txt"

    # --- 模型初始化 ---
    print("--- 正在初始化 LPIPS 模型 ---")
    lpips_model = lpips.LPIPS(net='vgg', spatial=False).to(DEVICE)
    lpips_model.eval() 

    print(f"\n--- 正在加载数据集 {PARQUET_FILE_PATH} ---")
    dataset_dict = load_dataset("parquet", data_files=PARQUET_FILE_PATH, split="train")

    # --- 0. 初始化 Pipeline (仅在生成阶段需要) ---
    if RUN_GENERATION_PHASE:
        sampler, adapter = initialize_basemodel_and_adapter(base_model_path=base_model_path,
                                                adapter_config_name=adapter_config_name,
                                                DEVICE=DEVICE
                                                )
        print("\n" + "="*50)
        print("                 阶段一：图像生成 (耗时) ")
        print("="*50)
        avg_structural_lpips_result = generation_phase(
            sampler, adapter, dataset_dict, TEMP_GEN_DIR, TEMP_REAL_DIR, TEMP_PROMPTS_FILE, TEMP_LPIPS_FILE, IMAGE_SIZE,
            caption_col, conditioning_image_col, real_image_col, 
            control_strength, num_inference_steps, NUM_SAMPLES_TO_EVALUATE, 
            DEVICE, lpips_model
        )
        
        # 释放 VRAM 中的 ControlNet 模型和 LPIPS 模型
        del sampler, adapter
        del lpips_model
        torch.cuda.empty_cache()


    # --- 运行阶段二：指标计算 (快速) ---
    print("\n" + "="*50)
    print("              阶段二：指标计算 (快速) ")
    print("="*50)
    
    # 如果跳过了生成阶段，lpips_model 仍然在内存中，我们需要先释放它（如果它存在）。
    if not RUN_GENERATION_PHASE and 'lpips_model' in locals():
         del lpips_model
         torch.cuda.empty_cache()
         
    clip_score_final, fid_value, kid_value, final_lpips_score = metric_phase(
        TEMP_GEN_DIR, TEMP_REAL_DIR, TEMP_PROMPTS_FILE, TEMP_LPIPS_FILE, DEVICE, 
        avg_structural_lpips_from_phase1=avg_structural_lpips_result
    )

    # 打印和保存结果
    print("\n==================================")
    print("     ControlNet 详细评估结果      ")
    print("==================================")
    print(f"CLIP Score (文本一致性, 越高越好): {clip_score_final:.4f}")
    print(f"FID Score (图像质量/多样性, 越低越好): {fid_value:.4f}")
    print(f"KID Score (图像质量/多样性, 越低越好): {kid_value:.4f}")
    print(f"结构 LPIPS (保真度, 越低越好):      {final_lpips_score:.4f}")
    print("==================================")

    with open(OUTPUT_FILE_NAME, 'w') as f:
        f.write("--- T2I-Adapter 评估结果 ---\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluation Samples: {NUM_SAMPLES_TO_EVALUATE}\n")
        f.write("-" * 30 + "\n")
        f.write(f"CLIP Score (越高越好): {clip_score_final:.4f}\n")
        f.write(f"FID Score (越低越好): {fid_value:.4f}\n")
        f.write(f"KID Score (越低越好): {kid_value:.4f}\n")
        f.write(f"结构 LPIPS (越低越好): {final_lpips_score:.4f}\n")
        f.write("-" * 30 + "\n")
        
    print(f"\n✅ 评估结果已成功保存到文件: {OUTPUT_FILE_NAME}")