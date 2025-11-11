import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd

loss_file= 'check_points/15000/training_history.pkl'

with open(loss_file, 'rb') as f:
    # 使用 pickle.load() 读取数据
    history_data = pickle.load(f)

#print(history_data)

np.random.seed(42) # 为了结果可复现
raw_loss_list = history_data['train_losses']

loss_series = pd.Series(raw_loss_list)
WINDOW_SIZE = 1000 # 平滑窗口大小
smoothed_loss = loss_series.rolling(window=WINDOW_SIZE, min_periods=1).mean()

DESIRED_POINTS = 1000 # 例如，希望图上只有 200 个点
total_steps = len(raw_loss_list)
if total_steps > DESIRED_POINTS:
    # 计算需要跳过的步长
    sample_step = total_steps // DESIRED_POINTS
    # 或者可以使用 np.linspace 来获取精确的索引
    sample_indices = np.linspace(0, total_steps - 1, DESIRED_POINTS, dtype=int)
else:
 # 如果总点数少于或等于期望点数，则不采样，绘制所有点
    sample_indices = np.arange(total_steps)
    sample_step = 1 # 方便理解，实际不会用到
sampled_steps = sample_indices
sampled_raw_loss = [raw_loss_list[i] for i in sample_indices]
sampled_smoothed_loss = [smoothed_loss[i] for i in sample_indices]
plt.figure(figsize=(12, 6))

if total_steps > DESIRED_POINTS: # 只有当采样发生时才绘制采样点
    plt.plot(sampled_steps, sampled_raw_loss,
             label='Raw Loss (Sampled)',
             color='lightgray',
             linestyle='--',
             alpha=0.6,
             marker='.', # 可以添加点标记来强调是采样点
             markersize=4)


plt.plot(sampled_steps, sampled_smoothed_loss,
        label=f'Smoothed Loss (Window={WINDOW_SIZE}, {len(sampled_steps)} points)',
         color='red',
        linewidth=2)

plt.title('Training Loss Curve (Sampled Points)', fontsize=16)
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('training_loss_curve.png')

