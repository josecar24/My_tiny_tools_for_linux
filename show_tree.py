import os
import argparse

def show_tree(path, prefix="", max_depth=4, current_depth=0):
    if current_depth >= max_depth:
        return
    
    items = []
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        print(f"{prefix}[Permission Denied]")
        return
    
    # 过滤掉不需要的文件和目录
    ignore_patterns = ['.git', '__pycache__', '.pyc', 'node_modules', '.DS_Store', '.vscode']
    items = [item for item in items if not any(pattern in item for pattern in ignore_patterns)]
    
    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        is_last = i == len(items) - 1
        
        if is_last:
            print(f"{prefix}└── {item}")
            new_prefix = prefix + "    "
        else:
            print(f"{prefix}├── {item}")
            new_prefix = prefix + "│   "
        
        if os.path.isdir(item_path):
            show_tree(item_path, new_prefix, max_depth, current_depth + 1)

def main():
    parser = argparse.ArgumentParser(description='显示目录树结构')
    parser.add_argument('path', nargs='?', default='.', 
                       help='要显示的目录路径 (默认: 当前目录)')
    parser.add_argument('-d', '--depth', type=int, default=4,
                       help='最大显示深度 (默认: 4)')
    parser.add_argument('-i', '--ignore', nargs='+', 
                       default=['.git', '__pycache__', '.pyc', 'node_modules', '.DS_Store', '.vscode'],
                       help='要忽略的文件/目录模式')
    parser.add_argument('--dirs-only', action='store_true',
                       help='只显示目录，不显示文件')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    target_path = os.path.abspath(args.path)
    
    if not os.path.exists(target_path):
        print(f"错误: 路径 '{target_path}' 不存在")
        return
    
    if not os.path.isdir(target_path):
        print(f"错误: '{target_path}' 不是一个目录")
        return
    
    print(f"{os.path.basename(target_path)}/")
    show_tree_with_options(target_path, max_depth=args.depth, 
                          ignore_patterns=args.ignore, dirs_only=args.dirs_only)

def show_tree_with_options(path, prefix="", max_depth=4, current_depth=0, 
                          ignore_patterns=None, dirs_only=False):
    if current_depth >= max_depth:
        return
    
    if ignore_patterns is None:
        ignore_patterns = ['.git', '__pycache__', '.pyc', 'node_modules', '.DS_Store', '.vscode']
    
    items = []
    try:
        all_items = sorted(os.listdir(path))
        # 过滤不需要的文件和目录
        items = [item for item in all_items if not any(pattern in item for pattern in ignore_patterns)]
        
        # 如果只显示目录，过滤掉文件
        if dirs_only:
            items = [item for item in items if os.path.isdir(os.path.join(path, item))]
        
    except PermissionError:
        print(f"{prefix}[Permission Denied]")
        return
    
    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        is_last = i == len(items) - 1
        is_dir = os.path.isdir(item_path)
        
        # 显示项目
        symbol = "└──" if is_last else "├──"
        display_name = f"{item}/" if is_dir else item
        print(f"{prefix}{symbol} {display_name}")
        
        # 递归处理子目录
        if is_dir:
            new_prefix = prefix + ("    " if is_last else "│   ")
            show_tree_with_options(item_path, new_prefix, max_depth, current_depth + 1,
                                 ignore_patterns, dirs_only)

if __name__ == "__main__":
    main()

# Example usage:

# # 显示当前目录
# python show_tree.py

# # 显示指定目录
# python show_tree.py /home/zitian.tang@beigenecorp.net/agent_trial

# # 显示指定目录，限制深度为2层
# python show_tree.py /home/zitian.tang@beigenecorp.net/agent_trial -d 2

# # 只显示目录结构，不显示文件
# python show_tree.py /home/zitian.tang@beigenecorp.net/agent_trial --dirs-only

# # 自定义忽略的文件/目录模式
# python show_tree.py /home/zitian.tang@beigenecorp.net/agent_trial -i .git __pycache__ *.log

# # 查看帮助
# python show_tree.py -h

