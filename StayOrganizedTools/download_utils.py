
import argparse
from tqdm import tqdm
import urllib.request



class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


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