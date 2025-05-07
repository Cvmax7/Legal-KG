import os
import shutil


def create_legal_corpus_structure(source_dir="."):
    """
    创建法律文本语料库目录结构

    参数:
    source_dir: str, 源文件所在目录，默认为当前目录
    """
    # 创建基础目录
    base_dir = "./legal_corpus"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)  # 如果目录存在，先删除
    os.makedirs(base_dir)

    # 记录处理结果
    processed_files = []
    missing_files = []

    # 遍历源目录中的所有目录
    for dir_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, dir_name)

        # 只处理目录
        if os.path.isdir(source_path):
            # 将目录名转换为英文分类名
            category_map = {
                "经济法": "economic_law",
                "民法典": "civil_law",
                "民法商法": "commercial_law",
                "其他": "other_law",
                "社会法": "social_law",
                "司法解释": "judicial_interpretation",
                "诉讼与非诉讼程序法": "procedure_law",
                "宪法": "constitutional_law",
                "宪法相关法": "constitutional_related_law",
                "刑法": "criminal_law",
                "行政法": "administrative_law",
                "行政法规": "administrative_regulation",
            }

            category = category_map.get(dir_name, f"other_{dir_name}")
            category_dir = os.path.join(base_dir, category)

            # 创建目标目录
            os.makedirs(category_dir, exist_ok=True)

            # 遍历目录中的所有.md文件
            for root, _, files in os.walk(source_path):
                for file in files:
                    if file.endswith(".md"):
                        source_file = os.path.join(root, file)
                        # 保持相对路径结构
                        rel_path = os.path.relpath(source_file, source_path)
                        dest_file = os.path.join(category_dir, rel_path)

                        # 确保目标目录存在
                        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

                        # 复制文件
                        shutil.copy2(source_file, dest_file)
                        processed_files.append(f"{category}/{rel_path}")
                        print(f"复制文件: {source_file} -> {dest_file}")

    # 打印处理结果
    print("\n=== 处理结果 ===")
    print(f"成功处理的文件: {len(processed_files)}")
    for file in processed_files:
        print(f"  - {file}")


if __name__ == "__main__":
    source_dir = "../Laws-master"  
    create_legal_corpus_structure(source_dir)
