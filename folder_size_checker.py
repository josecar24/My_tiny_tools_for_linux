#!/usr/bin/env python3
"""
A simple tool to check the size of any folder.
Usage: python folder_size_checker.py /path/to/folder
"""

import os
import argparse
import sys
from pathlib import Path


def get_folder_size(folder_path: str) -> int:
    """
    Calculate total size of a folder in bytes.
    
    Args:
        folder_path: Path to the folder
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(file_path)
                except (OSError, IOError) as e:
                    print(f"Warning: Could not access {file_path}: {e}", file=sys.stderr)
                    continue
    except (OSError, IOError) as e:
        print(f"Error: Could not access folder {folder_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    return total_size


def format_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Check the total size of a folder and its contents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python folder_size_checker.py /home/user/Documents
  python folder_size_checker.py . --bytes
  python folder_size_checker.py /var/log --verbose
        """
    )
    
    parser.add_argument(
        "path",
        help="Path to the folder to check"
    )
    
    parser.add_argument(
        "--bytes",
        action="store_true",
        help="Show size in bytes only (no human-readable format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show additional details like file count"
    )
    
    args = parser.parse_args()
    
    # Validate path
    folder_path = Path(args.path).resolve()
    
    if not folder_path.exists():
        print(f"Error: Path '{args.path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"Error: Path '{args.path}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    
    # Calculate size
    print(f"Calculating size of: {folder_path}")
    total_size = get_folder_size(str(folder_path))
    
    # Display results
    if args.bytes:
        print(f"Total size: {total_size} bytes")
    else:
        print(f"Total size: {format_size(total_size)} ({total_size:,} bytes)")
    
    # Verbose output
    if args.verbose:
        file_count = 0
        dir_count = 0
        
        try:
            for root, dirs, files in os.walk(folder_path):
                file_count += len(files)
                dir_count += len(dirs)
            
            print(f"Files: {file_count:,}")
            print(f"Directories: {dir_count:,}")
        except (OSError, IOError) as e:
            print(f"Warning: Could not count files/directories: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

# example user runs:
# python folder_size_checker.py /home/user/Documents --verbose
