# organize_dataset.py
import os
import shutil
import random
from tqdm import tqdm

# ================= 请修改以下路径为您的实际路径 =================
base_dir = r"path/to/your/raw/data"   # 改为您的原始数据根目录
train_img_dir = os.path.join(base_dir, "train")
label_dir = os.path.join(base_dir, "label")
output_dir = os.path.join(base_dir, "data_organized")

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
random_seed = 42
# ================================================

random.seed(random_seed)

# 获取所有患者ID
patient_ids = []
for folder in os.listdir(train_img_dir):
    folder_path = os.path.join(train_img_dir, folder)
    if os.path.isdir(folder_path):
        label_folder = os.path.join(label_dir, folder)
        if os.path.isdir(label_folder):
            patient_ids.append(folder)
        else:
            print(f"警告: 患者 {folder} 缺少对应的掩膜文件夹")

print(f"找到 {len(patient_ids)} 个有效患者")

random.shuffle(patient_ids)
n_train = int(len(patient_ids) * train_ratio)
n_val = int(len(patient_ids) * val_ratio)
train_ids = patient_ids[:n_train]
val_ids = patient_ids[n_train:n_train+n_val]
test_ids = patient_ids[n_train+n_val:]

print(f"训练集: {len(train_ids)} 患者")
print(f"验证集: {len(val_ids)} 患者")
print(f"测试集: {len(test_ids)} 患者")

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)

def copy_patient_images(patient_id, split):
    img_src_dir = os.path.join(train_img_dir, patient_id)
    mask_src_dir = os.path.join(label_dir, patient_id)
    img_dst_dir = os.path.join(output_dir, split, 'images')
    mask_dst_dir = os.path.join(output_dir, split, 'masks')
    
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    img_files = [f for f in os.listdir(img_src_dir) if f.lower().endswith(img_extensions)]
    
    for img_file in img_files:
        img_src = os.path.join(img_src_dir, img_file)
        # 尝试同名掩膜
        mask_file = img_file
        mask_src = os.path.join(mask_src_dir, mask_file)
        if not os.path.exists(mask_src):
            base = os.path.splitext(img_file)[0]
            mask_file = base + '.png'
            mask_src = os.path.join(mask_src_dir, mask_file)
        if not os.path.exists(mask_src):
            print(f"警告: 患者 {patient_id} 的图片 {img_file} 找不到对应掩膜，跳过")
            continue
        
        new_name = f"{patient_id}_{img_file}"
        new_mask_name = f"{patient_id}_{mask_file}"
        shutil.copy2(img_src, os.path.join(img_dst_dir, new_name))
        shutil.copy2(mask_src, os.path.join(mask_dst_dir, new_mask_name))

print("处理训练集...")
for pid in tqdm(train_ids):
    copy_patient_images(pid, 'train')
print("处理验证集...")
for pid in tqdm(val_ids):
    copy_patient_images(pid, 'val')
print("处理测试集...")
for pid in tqdm(test_ids):
    copy_patient_images(pid, 'test')

print(f"完成！输出目录: {output_dir}")