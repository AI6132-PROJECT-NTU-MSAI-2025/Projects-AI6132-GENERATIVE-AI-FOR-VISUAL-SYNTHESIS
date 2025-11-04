import pandas as pd
import os
import glob
from typing import List

INPUT_DIR = "E:\Python_projects\AI6132-GAI-DATA\huggingface_dog6K"
OUTPUT_BASE_NAME = "og6k_processed"  # 新增：分片输出文件的基础名称
OUTPUT_DIR = "E:\Python_projects\AI6132-GAI-DATA\dog6k_processed"  # 新增：分片输出的文件夹
NUM_SHARDS = 10

VALID_SIZE = 0.1
TEST_SIZE = 0.1
RANDOM_STATE = 42

print(f"输入目录: {INPUT_DIR}")
print(f"分片输出目录: {OUTPUT_DIR}")
print(f"分片数量 (N): {NUM_SHARDS}")
print("-" * 30)

def split_and_tag_parquet(
        input_filepath: str,
        valid_size: float,
        test_size: float,
        random_state: int
) -> pd.DataFrame | None:
    """
    读取单个 Parquet 文件，进行 8:1:1 随机切分，并添加 'split' 标记列。
    返回合并后的 DataFrame。
    """
    filename = os.path.basename(input_filepath)
    print(f"-> 正在处理文件: {filename}")

    try:
        df = pd.read_parquet(input_filepath)
        total_rows = len(df)
        print(f"   - 原始行数: {total_rows}")

        if total_rows == 0:
            print("   - 警告：文件为空，跳过处理。")
            return None

        # --- 1. 分离出测试集 (10%) ---
        test_df = df.sample(frac=test_size, random_state=random_state).copy()
        test_df['split'] = 'test'  # 添加标记列
        remaining_df = df.drop(test_df.index)

        # --- 2. 从剩余部分中分离出验证集 (占总体的 10%) ---
        valid_frac_of_remaining = valid_size / (1 - test_size)
        valid_df = remaining_df.sample(
            frac=valid_frac_of_remaining,
            random_state=random_state
        ).copy()
        valid_df['split'] = 'validation'  # 添加标记列

        # --- 3. 剩下的作为训练集 (80%) ---
        train_df = remaining_df.drop(valid_df.index).copy()
        train_df['split'] = 'train'  # 添加标记列

        # 检查切分是否正确
        final_len = len(train_df) + len(valid_df) + len(test_df)
        assert total_rows == final_len
        print(f"   - 训练/验证/测试行数: {len(train_df)} / {len(valid_df)} / {len(test_df)}")
        print("   - ✅ 切分检查通过。")

        # --- 4. 合并并返回 ---
        combined_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        return combined_df

    except Exception as e:
        print(f"   - 错误：处理 {filename} 时发生异常: {e}")
        return None


# --- 3. 批量处理和分片逻辑 ---
parquet_files = glob.glob(os.path.join(INPUT_DIR, '*.parquet'), recursive=False)

if not parquet_files:
    print(f"未在目录 {INPUT_DIR} 中找到任何 .parquet 文件。请检查路径。")
else:
    print(f"共找到 {len(parquet_files)} 个 .parquet 文件，开始批量处理...")

    all_data_frames = []

    # 循环处理找到的每个文件 (沿用)
    for i, file_path in enumerate(parquet_files):
        print(f"\n--- 文件 {i + 1}/{len(parquet_files)} ---")

        split_df = split_and_tag_parquet(
            input_filepath=file_path,
            valid_size=VALID_SIZE,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        if split_df is not None:
            all_data_frames.append(split_df)

    if all_data_frames:
        # 第一次合并：将所有输入文件的结果合并成一个大数据集
        final_combined_df = pd.concat(all_data_frames, ignore_index=True)
        total_rows = len(final_combined_df)

        print("\n--- 开始大数据集分片处理 ---")
        print(f"最终总行数: {total_rows}")

        # 创建分片输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 计算每个分片的大小（向上取整以确保最后一个分片不会丢失数据）
        shard_size = (total_rows + NUM_SHARDS - 1) // NUM_SHARDS

        # 执行分片和保存
        for i in range(NUM_SHARDS):
            start_index = i * shard_size
            end_index = min((i + 1) * shard_size, total_rows)

            # 提取分片数据
            shard_df = final_combined_df.iloc[start_index:end_index]

            # 构造分片文件名 (例如: combined_shard-00000-of-00008.parquet)
            output_filename = f"{OUTPUT_BASE_NAME}-{i:05}-of-{NUM_SHARDS:05}.parquet"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)

            # 保存分片
            shard_df.to_parquet(output_filepath, index=False)

            print(f"✅ 分片 {i + 1}/{NUM_SHARDS} ({len(shard_df)} 行) 已保存到: {output_filepath}")

        print("\n===============================")
        print(f"✨ 所有数据已成功分割成 {NUM_SHARDS} 份并保存到 {OUTPUT_DIR} 文件夹！")
        print(f"切分计数:\n{final_combined_df['split'].value_counts()}")
        print("===============================")
    else:
        print("\n没有成功处理任何数据，未生成输出文件。")