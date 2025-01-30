import os
import shutil
from sklearn.model_selection import train_test_split


def create_dataset(gc10_det_folder, output_folder):
    # 创建训练、验证和测试集的输出文件夹结构
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_folder, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, split, 'labels'), exist_ok=True)

        # 存储所有图像和标签
    all_images = []
    all_labels = []

    # 遍历 GC10-DET 中的每个子文件夹
    for folder in os.listdir(gc10_det_folder):
        folder_path = os.path.join(gc10_det_folder, folder)
        if os.path.isdir(folder_path):  # 确保是一个文件夹
            # 定义图像文件夹和标签文件夹
            images_dir = folder_path
            labels_dir = os.path.join(gc10_det_folder, 'txt')  # 假设标签在txt文件夹中

            # 决定对应的标签文件
            for img_file in os.listdir(images_dir):
                if img_file.endswith('.jpg'):  # 根据你的图片格式调整
                    # 无需检测对应标签文件，假设其存在
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    all_images.append(os.path.join(images_dir, img_file))
                    all_labels.append(os.path.join(labels_dir, label_file))

                    # 划分数据集
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42
    )

    # 保存图像和标签到对应的目录
    for split, img_list, lbl_list in zip(splits, [train_images, val_images, test_images],
                                         [train_labels, val_labels, test_labels]):
        for img, lbl in zip(img_list, lbl_list):
            # 复制图像文件
            shutil.copy(img, os.path.join(output_folder, split, 'images', os.path.basename(img)))
            # 复制标签文件
            shutil.copy(lbl, os.path.join(output_folder, split, 'labels', os.path.basename(lbl)))

        # 设置文件夹路径


gc10_det_folder = './GC10-DET'  # 静态数据集文件夹路径
output_folder = './GC10_yolo_new'  # 输出路径
create_dataset(gc10_det_folder, output_folder)

print("数据集划分完成！")