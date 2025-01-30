import os


def find_out_of_range_labels(base_dir="GC10_yolo_new"):
    """
    1. 遍历 base_dir 下 train、test、val 的 labels 文件夹，用于查找所有文本标签文件。
    2. 如果遇到类别为10，自动将其改为9并写回文件。
    3. 检测所有超出 0~9 (且不等于10) 的标签，若有则记录所在的文件名。
    4. 统计所有出现过的标签类别（含替换后）。
    """
    subfolders = ["train", "test", "val"]
    out_of_range_files = []  # 用于存放含超范围标签的文件路径
    all_label_classes = set()  # 用于统计所有出现过的标签类别

    for subfolder in subfolders:
        labels_dir = os.path.join(base_dir, subfolder, "labels")
        if not os.path.exists(labels_dir):
            continue

        for txt_file in os.listdir(labels_dir):
            if txt_file.endswith(".txt"):
                txt_path = os.path.join(labels_dir, txt_file)

                # 读取文件内容
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                    # 用于暂存修改后的行
                updated_lines = []
                # 记录该文件是否包含超范围（非10）标签
                file_has_out_of_range = False

                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        # 空行或异常行，保持原样
                        updated_lines.append(line.strip())
                        continue

                    label_class = int(parts[0])
                    # 如果标签是10，则改为9
                    if label_class == 10:
                        label_class = 9

                        # 将（修改/未修改的）标签类别加入统计集合
                    all_label_classes.add(label_class)

                    # 检查是否超出 0~9 范围
                    # 注意： 如果原先是10，则已被改为9，不会进这里
                    if label_class < 0 or label_class > 9:
                        file_has_out_of_range = True

                        # 替换parts[0]后再重新拼接
                    parts[0] = str(label_class)
                    updated_lines.append(" ".join(parts))

                    # 将更新后的内容写回文件（覆盖原文件）
                with open(txt_path, "w", encoding="utf-8") as f:
                    # 保持每行末尾有换行
                    f.write("\n".join(updated_lines) + "\n")

                    # 如果该文件存在超范围标签，则记录文件路径
                if file_has_out_of_range:
                    out_of_range_files.append(txt_path)

    return out_of_range_files, all_label_classes


if __name__ == "__main__":
    out_of_range_files, all_labels = find_out_of_range_labels("GC10_yolo_new")

    # 打印统计结果
    if out_of_range_files:
        print("以下文件包含不在0~9范围内(且不等于10)的类别标注：")
        for file_path in out_of_range_files:
            print(f"- {file_path}")
    else:
        print("未发现任何不在0~9范围内的类别标注。")

        # 统计并显示所有（经过替换后的）标签类别
    print(f"\n数据集中一共出现了 {len(all_labels)} 种不同的标签类别：")
    print(sorted(all_labels))

