import os
from PIL import Image

# 입력 폴더와 출력 폴더
input_dir = "/root/dataset/rec/gKR_sBM_cAll_d28_mKR1_ch07/images"
output_dir = os.path.join(input_dir, "square_cropped")
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(".png"):
        path = os.path.join(input_dir, fname)
        img = Image.open(path)

        w, h = img.size
        # 정사각 크기 = 세로길이
        square_size = h

        # 가로 기준 중앙 정렬
        left = (w - square_size) // 2
        top = 0
        right = left + square_size
        bottom = h

        cropped = img.crop((left, top, right, bottom))
        save_path = os.path.join(output_dir, fname)
        cropped.save(save_path)

print("✅ 세로 길이에 맞춰 중앙 정사각 크롭 완료. 저장 위치:", output_dir)
