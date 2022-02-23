import argparse
import shutil

from pathlib import Path
def main(args):
    suffix = args.suffix
    for path in Path(args.path).rglob('*.'+suffix):
        print(path)
        print(f"Removing the file {path}")
        # shutil.rmtree(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove files with a certain suffix in a path')

    parser.add_argument('--path', type=str, help='Path to remove the file')
    parser.add_argument('--suffix', type=str, help='Suffix of files to remove')
    args = parser.parse_args()
    main(args)
