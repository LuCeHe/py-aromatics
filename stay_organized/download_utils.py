import argparse
from tqdm import tqdm
import urllib.request
import os, sys, io
import numpy as np
from urllib.parse import urlparse
import tarfile
from zipfile import ZipFile
from google.cloud import storage

# this file path
FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', '..', 'data', ))


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):

    if 'storage.cloud.google' in url:
        # split output_path
        datadir = os.path.split(output_path)[0]
        download_blob(url, datadir)

    else:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def decode_gcs_url(url):
    p = urlparse(url)
    path = p.path[1:].split('/', 1)
    bucket, file_path = path[0], path[1]
    return bucket, file_path


def download_blob(url, folderpath=DATADIR):
    storage_client = storage.Client.create_anonymous_client()
    bucket, filename = decode_gcs_url(url)
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(filename)
    file_path = os.path.join(folderpath, filename)

    with open(file_path, 'wb') as f:
        with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
            storage_client.download_blob_to_file(blob, file_obj)


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


def download_and_unzip(data_links, destination_dir, unzip_what=None):
    if not isinstance(unzip_what, list):
        unzip_what = [unzip_what]
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
            if not unzip_what[0] == None:
                for tag in unzip_what:
                    with tarfile.open(destination) as tar:

                        for member in tar.getmembers():
                            if tag in member.name:
                                tar.extract(member, destination_dir)

            elif destination.endswith("tar.gz"):
                tar = tarfile.open(destination, "r:gz")
                tar.extractall(destination_dir)
                tar.close()

            elif destination.endswith(".gz"):
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

        if any([f in destination for f in ['.zip', '.tar', '.tgz', '.gz']]):
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

