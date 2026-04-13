import cv2
import numpy as np

# 1. 读取图像
image1 = cv2.imread('img/1.png')
image2 = cv2.imread('img/2.png')

if image1 is None or image2 is None:
    raise ValueError("One of the images is not loaded correctly")

# 2. 检测ORB特征点和描述符
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# 3. 使用 BFMatcher 进行特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 4. Lowe's ratio test 过滤不可靠的匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 5. 计算优秀匹配点数量及匹配率
num_good_matches = len(good_matches)
num_keypoints = min(len(keypoints1), len(keypoints2))
match_ratio = num_good_matches / num_keypoints

print(f"Number of good matches: {num_good_matches}")
print(f"Match ratio: {match_ratio:.2f}")

# 6. 绘制并显示匹配结果，仅显示优秀匹配点
result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 7. 显示并保存结果图像
cv2.imshow('Good Matches', result_image)
cv2.imwrite('good_matches_orb.jpg', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
