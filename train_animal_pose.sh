#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 HF_HOME="/hy-tmp/hg_cache" \
accelerate launch train_animal_pose.py \
--pretrained_model_name_or_path \
stabilityai/stable-diffusion-xl-base-1.0 \
--output_dir experiments/adapter_pose_xl_debug \
--config configs/train/Adapter-XL-pose.yaml \
--mixed_precision="fp16" \
--resolution=1024 \
--learning_rate=1e-5 \
--max_train_steps=60000 \
--train_batch_size=2 \
--gradient_accumulation_steps=4 \
--report_to="tensorboard" \
--seed=42 \
--num_train_epochs 100 \
--logging_dir "/hy-tmp/Projects-AI6132-GENERATIVE-AI-FOR-VISUAL-SYNTHESIS/logs" \
--resume_from_checkpoint "experiments/adapter_sketch_xl/checkpoint-3000"