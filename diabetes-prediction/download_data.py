"""
从Kaggle下载糖尿病预测竞赛数据

使用前需要配置Kaggle API：
1. 登录Kaggle账户
2. 进入 Account -> API -> Create New API Token
3. 下载kaggle.json文件
4. 将kaggle.json放到 ~/.kaggle/ 目录
5. 设置权限：chmod 600 ~/.kaggle/kaggle.json

常见的糖尿病预测竞赛：
- pima-indians-diabetes-database
- diabetes-prediction-dataset
"""

import os
import sys
import zipfile
import subprocess


def check_kaggle_installed():
    """检查是否安装了kaggle包"""
    try:
        import kaggle
        return True
    except ImportError:
        return False


def install_kaggle():
    """安装kaggle包"""
    print("正在安装kaggle包...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
    print("kaggle包安装完成！")


def check_kaggle_credentials():
    """检查Kaggle API凭证是否配置"""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("\n" + "=" * 70)
        print("错误：未找到Kaggle API凭证")
        print("=" * 70)
        print("\n请按以下步骤配置Kaggle API：")
        print("\n1. 登录Kaggle账户: https://www.kaggle.com/")
        print("2. 进入 Account Settings -> API")
        print("3. 点击 'Create New API Token'")
        print("4. 下载 kaggle.json 文件")
        print("5. 将文件移动到 ~/.kaggle/ 目录:")
        print(f"   mkdir -p {kaggle_dir}")
        print(f"   mv ~/Downloads/kaggle.json {kaggle_dir}/")
        print(f"   chmod 600 {kaggle_json}")
        print("\n" + "=" * 70)
        return False
    
    return True


def list_available_datasets():
    """列出可用的糖尿病预测数据集"""
    print("\n" + "=" * 70)
    print("搜索Kaggle上的糖尿病预测数据集...")
    print("=" * 70)
    
    try:
        import kaggle
        # 搜索糖尿病相关数据集
        datasets = kaggle.api.dataset_list(search='diabetes prediction')
        
        print("\n找到以下糖尿病预测相关数据集：\n")
        for i, dataset in enumerate(datasets[:10], 1):
            print(f"{i}. {dataset.ref}")
            print(f"   标题: {dataset.title}")
            print(f"   大小: {dataset.size}")
            print(f"   下载次数: {dataset.downloadCount}")
            print()
        
        return [dataset.ref for dataset in datasets[:10]]
    except Exception as e:
        print(f"搜索失败: {e}")
        return []


def download_dataset(dataset_ref, download_path='data'):
    """
    下载指定的Kaggle数据集
    
    Parameters:
    -----------
    dataset_ref : str
        数据集引用，格式：username/dataset-name
    download_path : str
        下载路径
    """
    try:
        import kaggle
        
        print(f"\n正在下载数据集: {dataset_ref}")
        print(f"下载位置: {download_path}")
        
        # 创建下载目录
        os.makedirs(download_path, exist_ok=True)
        
        # 下载数据集
        kaggle.api.dataset_download_files(
            dataset_ref,
            path=download_path,
            unzip=True
        )
        
        print(f"\n✅ 数据集下载完成！")
        print(f"\n下载的文件:")
        for file in os.listdir(download_path):
            file_path = os.path.join(download_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def download_competition(competition_name, download_path='data'):
    """
    下载Kaggle竞赛数据
    
    Parameters:
    -----------
    competition_name : str
        竞赛名称
    download_path : str
        下载路径
    """
    try:
        import kaggle
        
        print(f"\n正在下载竞赛数据: {competition_name}")
        print(f"下载位置: {download_path}")
        
        # 创建下载目录
        os.makedirs(download_path, exist_ok=True)
        
        # 下载竞赛文件
        kaggle.api.competition_download_files(
            competition_name,
            path=download_path
        )
        
        # 解压文件
        zip_file = os.path.join(download_path, f"{competition_name}.zip")
        if os.path.exists(zip_file):
            print("\n正在解压文件...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(zip_file)
            print("解压完成！")
        
        print(f"\n✅ 竞赛数据下载完成！")
        print(f"\n下载的文件:")
        for file in os.listdir(download_path):
            file_path = os.path.join(download_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因：")
        print("1. 竞赛名称不正确")
        print("2. 需要先在Kaggle网站上接受竞赛规则")
        print("3. 没有权限访问该竞赛")
        return False


def main():
    """主函数"""
    print("=" * 70)
    print("Kaggle糖尿病预测数据集下载工具")
    print("=" * 70)
    
    # 检查并安装kaggle包
    if not check_kaggle_installed():
        print("\n未安装kaggle包，正在安装...")
        install_kaggle()
    
    # 检查API凭证
    if not check_kaggle_credentials():
        return
    
    print("\n请选择下载方式:")
    print("1. 搜索并下载数据集")
    print("2. 直接输入数据集名称")
    print("3. 下载竞赛数据")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == '1':
        # 搜索数据集
        datasets = list_available_datasets()
        if not datasets:
            print("\n未找到相关数据集。")
            return
        
        selection = input("\n请输入要下载的数据集编号 (1-10): ").strip()
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(datasets):
                download_dataset(datasets[idx])
            else:
                print("无效的编号！")
        except ValueError:
            print("请输入有效的数字！")
    
    elif choice == '2':
        # 直接输入数据集名称
        print("\n常见的糖尿病数据集:")
        print("  - uciml/pima-indians-diabetes-database")
        print("  - iammustafatz/diabetes-prediction-dataset")
        print("  - mathchi/diabetes-data-set")
        
        dataset_ref = input("\n请输入数据集名称 (格式: username/dataset-name): ").strip()
        if dataset_ref:
            download_dataset(dataset_ref)
        else:
            print("数据集名称不能为空！")
    
    elif choice == '3':
        # 下载竞赛数据
        print("\n注意：下载竞赛数据前，您需要：")
        print("1. 在Kaggle网站上找到对应的竞赛")
        print("2. 点击'Join Competition'并接受规则")
        
        competition = input("\n请输入竞赛名称: ").strip()
        if competition:
            download_competition(competition)
        else:
            print("竞赛名称不能为空！")
    
    else:
        print("无效的选择！")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()









