import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress  # 导入线性回归函数

loss_file = 'check_points/checkpoint-1000/'

with open(loss_file, 'rb') as f:
    history_data = pickle.load(f)

np.random.seed(42)
raw_loss_list = history_data['train_losses']

loss_series = pd.Series(raw_loss_list)
WINDOW_SIZE = 500
smoothed_loss = loss_series.rolling(window=WINDOW_SIZE, min_periods=1).mean()

DESIRED_POINTS = 1000
total_steps = len(raw_loss_list)
if total_steps > DESIRED_POINTS:
    sample_indices = np.linspace(0, total_steps - 1, DESIRED_POINTS, dtype=int)
else:
    sample_indices = np.arange(total_steps)

sampled_steps = sample_indices
sampled_raw_loss = [raw_loss_list[i] for i in sample_indices]
sampled_smoothed_loss = [smoothed_loss[i] for i in sample_indices]

plt.figure(figsize=(12, 6))

plt.plot(sampled_steps, sampled_smoothed_loss,
         label=f'Smoothed Loss (Window={WINDOW_SIZE}, {len(sampled_steps)} points)',
         color='red',
         linewidth=2)


# --- 添加线性拟合曲线的函数 ---
def add_linear_fit(ax, all_steps, all_loss_values, start_step, end_step, label_suffix=""):
    """
    在指定轴上添加线性拟合曲线。
    ax: matplotlib Axes 对象
    all_steps: 所有训练步数的列表或数组
    all_loss_values: 对应所有损失值的列表或数组
    start_step: 拟合的起始训练步数
    end_step: 拟合的结束训练步数
    label_suffix: 标签后缀，用于区分不同的拟合线
    """
    # 筛选出指定范围内的步数和损失值
    fit_steps = []
    fit_loss = []
    for i, step in enumerate(all_steps):
        if start_step <= step <= end_step:
            fit_steps.append(step)
            fit_loss.append(all_loss_values[i])

    if len(fit_steps) < 2:
        print(f"警告: 范围 [{start_step}, {end_step}] 内数据点不足2个，无法进行线性拟合。")
        return

    # 执行线性回归
    slope, intercept, r_value, p_value, std_err = linregress(fit_steps, fit_loss)

    # 生成拟合曲线的点
    # 为了避免拟合线只在采样点上显示，我们可以在原始的拟合范围内生成更多点
    x_fit = np.linspace(min(fit_steps), max(fit_steps), 100)  # 生成100个点来绘制拟合线
    y_fit = slope * x_fit + intercept

    ax.plot(x_fit, y_fit,
            label=f'Linear Fit ({start_step}-{end_step} steps{label_suffix})',
            color='blue',  # 可以根据需要选择颜色
            linestyle='--',
            linewidth=2)

    print(f"拟合范围: [{start_step}, {end_step}] - 斜率: {slope:.6f}, 截距: {intercept:.4f}")


# --- 调用 add_linear_fit 函数来添加拟合曲线 ---
# 您可以多次调用此函数来添加不同范围的拟合线

# 示例 1: 对整个平滑曲线进行线性拟合
# add_linear_fit(plt.gca(), sampled_steps, sampled_smoothed_loss,
#                start_step=min(sampled_steps), end_step=max(sampled_steps),
#                label_suffix=" (Full Range)")

# 示例 2: 拟合早期阶段的损失 (例如 0 到 20000 步)
add_linear_fit(plt.gca(), sampled_steps, sampled_smoothed_loss,
               start_step=100, end_step=500,
               label_suffix=" (Early Stage)")

# 示例 3: 拟合后期阶段的损失 (例如 40000 到 80000 步)
add_linear_fit(plt.gca(), sampled_steps, sampled_smoothed_loss,
               start_step=500, end_step=20000,
               label_suffix=" (Late Stage)")

plt.title('Training Loss Curve with Linear Fits', fontsize=16)
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('training_loss_curve_with_linear_fits.png')
# plt.show() # 如果需要在显示窗口中查看