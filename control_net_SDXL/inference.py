import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset, Image as daIMG

# --- 配置 ---
base_model_path = "./SDXL_base_fp16"
controlnet_path = "check_points/24000"
control_strength = 0.7
num_inference_steps = 20
NUM_IMAGES_TO_GENERATE = 2 # <--- 批量生成的图片数量（不包含参考图）

# 图像和文本描述
conditioning_image = Image.open("335.png").convert("RGB") # 确保是 RGB 格式
caption = "a dog standing in the grass"

# --- 1. 加载 ControlNet 模型 ---
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

# --- 2. 创建推理 Pipeline (无 ControlNet) ---
print("--- 正在加载无 ControlNet 的 Pipeline ---")
pipe_without_controlnet = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe_without_controlnet.scheduler = UniPCMultistepScheduler.from_config(pipe_without_controlnet.scheduler.config)
pipe_without_controlnet.enable_model_cpu_offload()

# --- 3. 创建推理 Pipeline (有 ControlNet) ---
print("--- 正在加载有 ControlNet 的 Pipeline ---")
pipe_with_controlnet = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe_with_controlnet.scheduler = UniPCMultistepScheduler.from_config(pipe_with_controlnet.scheduler.config)
pipe_with_controlnet.enable_model_cpu_offload()

# --- 4. 批量生成无 ControlNet 的图片 ---
print(f"--- 开始批量生成 {NUM_IMAGES_TO_GENERATE} 张无 ControlNet 的图片 ---")
generated_images_without_control = pipe_without_controlnet(
    caption,
    num_inference_steps=num_inference_steps,
    num_images_per_prompt=NUM_IMAGES_TO_GENERATE,
    generator=torch.Generator("cuda").manual_seed(42)
).images

# --- 5. 批量生成有 ControlNet 的图片 ---
print(f"--- 开始批量生成 {NUM_IMAGES_TO_GENERATE} 张有 ControlNet 的图片 ---")
generated_images_with_control = pipe_with_controlnet(
    caption,
    image=conditioning_image, # <--- 传入 conditioning_image
    num_inference_steps=num_inference_steps,
    controlnet_conditioning_scale=control_strength,
    num_images_per_prompt=NUM_IMAGES_TO_GENERATE,
    generator=torch.Generator("cuda").manual_seed(42)
).images

print("--- 图片生成完成，开始合并 ---")

# --- 6. 图像合并逻辑 ---

# 整理第一行图片：[参考图, 无 ControlNet 生成图1, 无 ControlNet 生成图2, ...]
row1_images = [conditioning_image] + generated_images_without_control
# 整理第二行图片：[参考图, 有 ControlNet 生成图1, 有 ControlNet 生成图2, ...]
# 注意：这里我们重复使用参考图作为每行的第一个元素，以便对齐显示
row2_images = [conditioning_image] + generated_images_with_control

# 检查图片尺寸（假设所有图片尺寸一致，以参考图为准）
if not row1_images or not row2_images:
    raise ValueError("没有图片可供合并。")

img_width, img_height = row1_images[0].size
images_per_row = len(row1_images) # 每行的图片数量 (参考图 + 生成图)

# 创建一个新的画布，两行排列
# 宽度 = 图片宽度 * 每行图片数
# 高度 = 图片高度 * 2 (两行)
merged_width = img_width * images_per_row
merged_height = img_height * 2

# 新建一张白色背景的画布
merged_image = Image.new('RGB', (merged_width, merged_height), 'white')

# 字体设置
try:
    font = ImageFont.truetype("arial.ttf", 30) # 稍微调小字体
except IOError:
    font = ImageFont.load_default()

# 绘制第一行 (无 ControlNet)
y_offset = 0
draw = ImageDraw.Draw(merged_image)
draw.text((10, y_offset + 10), "Without ControlNet:", fill='red', font=font) # 标题
for i, img in enumerate(row1_images):
    img = img.resize((img_width, img_height)) # 确保尺寸匹配
    merged_image.paste(img, (img_width * i, y_offset))

    label = "Reference" if i == 0 else f"Output {i}"
    text_width, text_height = draw.textsize(label, font=font)
    text_x = img_width * i + (img_width - text_width) // 2
    text_y = y_offset + img_height - text_height - 10 # 放在图片下方
    draw.text((text_x, text_y), label, fill='black', font=font)


# 绘制第二行 (有 ControlNet)
y_offset = img_height # 第二行的 Y 偏移量
draw.text((10, y_offset + 10), "With ControlNet:", fill='red', font=font) # 标题
for i, img in enumerate(row2_images):
    img = img.resize((img_width, img_height)) # 确保尺寸匹配
    merged_image.paste(img, (img_width * i, y_offset))

    label = "Reference" if i == 0 else f"Output {i}"
    text_width, text_height = draw.textsize(label, font=font)
    text_x = img_width * i + (img_width - text_width) // 2
    text_y = y_offset + img_height - text_height - 10 # 放在图片下方
    draw.text((text_x, text_y), label, fill='black', font=font)

merged_image.save("merged_controlnet_comparison.png")
# 7. 显示或保存合并后的图片
merged_image.show()


print("--- 合并后的对比图片已显示并保存 ---")