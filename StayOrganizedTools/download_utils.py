import argparse
from tqdm import tqdm
import urllib.request
import os, sys, io
import numpy as np

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        io.FileIO.__init__(self, path, *args, **kwargs)

    def read(self, size):
        percentage = np.array(self.tell() / self._total_size * 100).round(3)
        percentage_string = str(percentage * 1e3)
        if percentage_string[-2] == '5':
            sys.stdout.write("Decompress progress: {}% \r".format(percentage))
            sys.stdout.flush()
        return io.FileIO.read(self, size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=None,
        type=str,
        required=True,
        help="url from where to download",
    )
    parser.add_argument(
        "--path",
        default=None,
        type=str,
        required=True,
        help="path where to save content of url",
    )
    args = parser.parse_args()
    download_url(args.url, args.path)
