import math
# 使用以下命令安装Pillow库：pip install Pillow
from PIL import Image

def token_calculate(image_path):
    # 打开指定的PNG图片文件
    image = Image.open(image_path)

    # 获取图片的原始尺寸
    height = image.height
    width = image.width
    
    # Qwen3-VL模型：将宽高都调整为32的整数倍
    # 其余模型：将宽高都调整为28的整数倍
    h_bar = round(height / 32) * 32 
    w_bar = round(width / 32) * 32
    
    # 图像的Token下限：4个Token
    min_pixels = 32 * 32 * 4
    # 图像的Token上限：1280个Token
    max_pixels = 1280 * 32 * 32
        
    # 对图像进行缩放处理，调整像素的总数在范围[min_pixels,max_pixels]内
    if h_bar * w_bar > max_pixels:
        # 计算缩放因子beta，使得缩放后的图像总像素数不超过max_pixels
        beta = math.sqrt((height * width) / max_pixels)
        # 重新计算调整后的高度，对于Qwen3-VL，确保为32的整数倍，对于其他模型，确保为28的整数倍
        h_bar = math.floor(height / beta / 32) * 32
        # 重新计算调整后的宽度，对于Qwen3-VL，确保为32的整数倍，对于其他模型，确保为28的整数倍
        w_bar = math.floor(width / beta / 32) * 32
    elif h_bar * w_bar < min_pixels:
        # 计算缩放因子beta，使得缩放后的图像总像素数不低于min_pixels
        beta = math.sqrt(min_pixels / (height * width))
        # 重新计算调整后的高度，对于Qwen3-VL，确保为32的整数倍，对于其他模型，确保为28的整数倍
        h_bar = math.ceil(height * beta / 32) * 32
        # 重新计算调整后的宽度，对于Qwen3-VL，确保为32的整数倍，对于其他模型，确保为28的整数倍，
        w_bar = math.ceil(width * beta / 32) * 32
    return h_bar, w_bar

# 将test.png替换为本地的图像路径
h_bar, w_bar = token_calculate("img\\20230505043842_2.jpeg")
print(f"缩放后的图像尺寸为：高度为{h_bar}，宽度为{w_bar}")

# 计算图像的Token数：对于Qwen3-VL，Token数 = 总像素除以32 * 32，对于其他模型，Token数 = 总像素除以28 * 28
token = int((h_bar * w_bar) / (32 * 32))

# 系统会自动添加<|vision_bos|>和<|vision_eos|>视觉标记（各计1个Token）
print(f"图像的Token数为{token + 2}")