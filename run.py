#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os

def convert_relative_paths() -> None:
    """
    Chuyển đổi các đường dẫn được truyền qua các tham số
    -s/--source, -t/--target, -o/--output thành đường dẫn tuyệt đối.
    """
    flags = ["-s", "--source", "-t", "--target", "-o", "--output"]
    cwd: str = os.getcwd()
    for i, arg in enumerate(sys.argv):
        if arg in flags and i + 1 < len(sys.argv):
            path: str = sys.argv[i + 1]
            if not os.path.isabs(path):
                sys.argv[i + 1] = os.path.abspath(path)

def main() -> None:
    convert_relative_paths()
    from roop import core
    core.run()

if __name__ == '__main__':
    main()
