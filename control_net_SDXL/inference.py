from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image

base_model_path = "./SDXL_base_fp16"
controlnet_path = "./check_points"

# 3. 从 'check_points' 文件夹加载 ControlNet 模型
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

# 4. 创建推理 Pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload() # 启用 offload 以节省 VRAM

conditioning_image = Image.open("27.png")
image = pipe(
     "A high-quality image of a dog, high detail",
     image=conditioning_image,
     num_inference_steps=20
 ).images[0]

image.show()