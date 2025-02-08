#!/usr/bin/env python3
import os
import sys

def convert_relative_paths() -> None:
    """
    Chuyển đổi các đường dẫn được truyền qua các tham số
    -s/--source, -t/--target, -o/--output thành đường dẫn tuyệt đối.
    """
    # Danh sách các flag cần xử lý
    flags = ["-s", "--source", "-t", "--target", "-o", "--output"]
    cwd: str = os.getcwd()
    # Duyệt qua các tham số trong sys.argv
    for i, arg in enumerate(sys.argv):
        if arg in flags and i + 1 < len(sys.argv):
            path: str = sys.argv[i + 1]
            if not os.path.isabs(path):
                sys.argv[i + 1] = os.path.abspath(path)

def main() -> None:
    """
    Hàm main() thực hiện chuyển đổi đường dẫn sau đó gọi core.run()
    """
    convert_relative_paths()
    # Import module core sau khi xử lý đường dẫn
    from roop import core
    core.run()

if __name__ == '__main__':
    main()
