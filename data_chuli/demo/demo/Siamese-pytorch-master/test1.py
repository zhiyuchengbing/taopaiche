import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ===============================
# 1. 加载预训练模型（去掉最后分类层）
# ===============================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉分类层
model.eval()  # 推理模式，不训练

# ===============================
# 2. 定义特征提取函数
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # 增加批次维度 [1, 3, 224, 224]
    with torch.no_grad():
        feat = model(img).squeeze()  # 输出为 2048维特征
    return feat

# ===============================
# 3. 提取两张车图的特征向量
# ===============================
car1_feat = extract_features("datasets/truck_only/22_20250926_184433_CH01_0001.jpg")
car2_feat = extract_features("datasets/truck_only/22_20250926_190113_CH01_0002.jpg")

# ===============================
# 4. 计算余弦相似度
# ===============================
similarity = F.cosine_similarity(car1_feat, car2_feat, dim=0).item()
print(f"两辆车外观相似度: {similarity:.4f}")

# ===============================
# 5. 判定是否疑似套牌
# ===============================
if similarity < 0.8:
    print("⚠️ 疑似套牌车：车牌相同但车型差异明显！")
else:
    print("✅ 同一辆车或极其相似的车型。")
