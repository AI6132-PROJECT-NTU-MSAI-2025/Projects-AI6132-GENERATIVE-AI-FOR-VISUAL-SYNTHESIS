from datasets import load_dataset
import os

# --- 配置您的路径 ---
# 包含所有 Parquet 文件的文件夹路径
PARQUET_FOLDER_PATH = "E:\Python_projects\AI6132-GAI-DATA\dog6k_processed"
# 输出的单个 Parquet 文件名
OUTPUT_PARQUET_PATH = "combined_test_data.parquet"


# --------------------

def extract_and_save_splits():
    """
    加载文件夹中的所有 Parquet 文件，过滤出 'test' 和 'evaluation' 部分，
    并保存为一个新的 Parquet 文件。
    """
    print(f"--- 1. 加载 Parquet 文件 ---")

    # 1. 加载文件夹中的所有 Parquet 文件
    # load_dataset 通常会将文件夹中的所有文件合并为一个默认的 split，
    # 我们可以通过 `data_files` 指定所有文件路径。
    try:
        # 使用 glob 模式加载文件夹中的所有 .parquet 文件
        dataset_dict = load_dataset("parquet",data_files=f"{PARQUET_FOLDER_PATH}/*.parquet")
        # 假设所有数据都被合并到了 'train' split 中 (这是 load_dataset 的默认行为)
        full_dataset = dataset_dict['train']
        print(f"成功加载，总行数: {len(full_dataset)}")

    except Exception as e:
        print(f"错误：加载数据集失败。请检查路径和文件格式。错误信息: {e}")
        return

    print(f"\n--- 2. 过滤数据集 ---")
    # 2. 定义过滤函数：保留 'split' 列为 'test' 或 'evaluation' 的行
    def is_target_split(example):
        """检查 'split' 字段是否在目标列表内"""
        # 确保 'split' 列存在
        if "split" not in example:
            return False
            # 检查是否为 'test' 或 'evaluation' (忽略大小写)
        return str(example["split"]).lower() in ["test", "validation"]

    # 应用过滤
    filtered_dataset = full_dataset.filter(is_target_split, num_proc=1)

    print(f"过滤完成。新的数据集行数: {len(filtered_dataset)}")

    # 3. 将过滤后的数据集保存为新的 Parquet 文件
    if len(filtered_dataset) > 0:
        print(f"\n--- 3. 保存到 {OUTPUT_PARQUET_PATH} ---")
        # 使用 to_parquet() 方法保存为一个新的 Parquet 文件
        filtered_dataset.to_parquet(OUTPUT_PARQUET_PATH)

        print(f"\ 任务完成！新的 Parquet 文件已保存到: {os.path.abspath(OUTPUT_PARQUET_PATH)}")
    else:
        print("\️ 警告：过滤后的数据集为空，未生成新的文件。请检查 'split' 列的值。")


if __name__ == "__main__":
    extract_and_save_splits()