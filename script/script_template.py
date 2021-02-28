import os
import gzip
import base64
from pathlib import Path
from typing import Dict


def main():
    # this is base64 encoded source code
    file_data: Dict = {file_data}

    for path, encoded in file_data.items():
        print(path)
        path = Path(path)
        os.makedirs(str(path.parent), exist_ok=True)
        path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


if __name__ == '__main__':
    main()
