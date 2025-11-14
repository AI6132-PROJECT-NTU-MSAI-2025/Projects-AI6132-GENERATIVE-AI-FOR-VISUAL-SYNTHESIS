import pytest
from PIL import Image
import numpy as np


# --- 待测试的函数 ---
# 假设你的函数位于这个文件的顶部或者被导入
def re_extract_condition_map(image: Image.Image, controlnet_type: str, pose_processor) -> Image.Image:
    """从生成的图像中重新提取条件图。"""
    if controlnet_type == "openpose":
        # 运行姿态估计器
        # OpenposeDetector 返回的图像已经是 PIL Image
        extracted_map = pose_processor(image.convert("RGB"), include_face=True, include_hand=True)
        return extracted_map
    else:
        # 如果不是 openpose，返回 RGB 图像
        return image.convert("RGB")


# --- 模拟 OpenPose 处理器 ---
# 我们需要一个模拟对象来替代实际的 OpenPose 模型，以便在测试中不依赖于实际的模型加载和运行。
class MockPoseProcessor:
    """模拟 OpenPose 处理器，返回一个占位图。"""

    def __init__(self, expected_size):
        self.expected_size = expected_size

    def __call__(self, image: Image.Image, include_face=False, include_hand=False) -> Image.Image:
        # 验证输入是否被转换为 RGB
        assert image.mode == "RGB"
        # 模拟 OpenPose 的输出，返回一个与输入同大小的黑色图片
        # 实际的 OpenPose 会返回骨架图
        return Image.new("RGB", self.expected_size, color="black")


# --- Pytest 测试函数 ---

# 准备一个共享的测试图片 fixture
@pytest.fixture
def test_image() -> Image.Image:
    """创建一个 128x128 的示例 PIL 图像。"""
    return Image.new("RGB", (128, 128), color="red")


# 测试 openpose 模式
def test_re_extract_openpose(test_image: Image.Image):
    # 1. 设置
    width, height = test_image.size
    mock_processor = MockPoseProcessor((width, height))
    controlnet_type = "openpose"

    # 2. 调用待测函数
    extracted_map = re_extract_condition_map(test_image, controlnet_type, mock_processor)

    # 3. 断言 (验证结果)
    # 检查返回的是否是一个 PIL Image
    assert isinstance(extracted_map, Image.Image)
    # 检查返回图像的尺寸是否与输入图像相同
    assert extracted_map.size == test_image.size
    # 检查返回图像的模式是否是 RGB
    assert extracted_map.mode == "RGB"

    # 根据 MockPoseProcessor 的逻辑，返回的应该是纯黑色图像
    assert np.all(np.array(extracted_map) == 0)
    print("\n✅ Openpose 模式测试通过：成功调用姿态处理器并返回模拟的条件图。")


# 测试非 openpose 模式
@pytest.mark.parametrize("controlnet_type", ["canny", "depth", "scribble", "tile"])
def test_re_extract_non_openpose(test_image: Image.Image, controlnet_type: str):
    # 1. 设置
    # 在非 openpose 模式下，pose_processor 不会被使用，可以传入 None 或 Mock
    mock_processor = MockPoseProcessor(test_image.size)

    # 2. 调用待测函数
    extracted_map = re_extract_condition_map(test_image, controlnet_type, mock_processor)

    # 3. 断言 (验证结果)
    # 检查返回的是否是一个 PIL Image
    assert isinstance(extracted_map, Image.Image)
    # 检查返回图像的尺寸是否与输入图像相同
    assert extracted_map.size == test_image.size
    # 检查返回图像的模式是否是 RGB (因为函数要求返回 RGB 图像)
    assert extracted_map.mode == "RGB"

    # 检查返回的图像是否就是输入图像的 RGB 副本 (纯红色)
    input_array = np.array(test_image)
    output_array = np.array(extracted_map)
    # 检查所有像素是否都是红色 (255, 0, 0)
    assert np.all(output_array == input_array)
    print(f"\n✅ 非 {controlnet_type} 模式测试通过：成功返回了原始图像的 RGB 副本。")