import argparse
from tqdm import tqdm
import urllib.request
import os, sys, io
import numpy as np
import os
import tarfile
from zipfile import ZipFile


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


def download_and_unzip(data_links, destination_dir):
    if not isinstance(data_links, list):
        data_links = [data_links]

    for origin in data_links:
        tail = os.path.split(origin)[1]
        destination = os.path.join(destination_dir, tail)

        desitination_name = destination.replace('.zip', '').replace('-v1', '').replace('.tgz', '').replace('.tar', '')
        is_file = os.path.isfile(desitination_name)
        is_folder = os.path.isdir(desitination_name)

        if not is_folder:
            if not os.path.isfile(destination): download_url(origin, destination)

            # Create a ZipFile Object and load sample.zip in it

            if destination.endswith("tar.gz"):
                tar = tarfile.open(destination, "r:gz")
                tar.extractall(destination_dir)
                tar.close()
            elif destination.endswith("tgz"):
                tar = tarfile.open(destination, "r:gz")
                tar.extractall(destination_dir)
                tar.close()
            elif destination.endswith("tar"):
                tar = tarfile.open(destination, "r:")
                tar.extractall(destination_dir)
                tar.close()
            elif destination.endswith("zip"):
                with ZipFile(destination, 'r') as zipObj:
                    # Extract all the contents of zip file in different directory
                    zipObj.extractall(destination_dir)

        if any([f in destination for f in ['.zip', '.tar', '.tgz']] ):
            try:
                os.remove(destination)
            except Exception as e:
                print(e)


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
