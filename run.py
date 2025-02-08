#!/usr/bin/env python3
import os
import sys

def convert_relative_paths():
    """
    Chuyển đổi các đường dẫn được truyền qua các tham số
    -s/--source, -t/--target, -o/--output thành đường dẫn tuyệt đối.
    """
    # Danh sách các flag cần xử lý
    flags = ["-s", "--source", "-t", "--target", "-o", "--output"]
    cwd = os.getcwd()
    # Duyệt qua các tham số trong sys.argv
    for i, arg in enumerate(sys.argv):
        if arg in flags and i + 1 < len(sys.argv):
            path = sys.argv[i + 1]
            if not os.path.isabs(path):
                sys.argv[i + 1] = os.path.abspath(path)

# Chuyển đổi các đường dẫn trước khi gọi core.run()
convert_relative_paths()

from roop import core

if __name__ == '__main__':
    core.run()
