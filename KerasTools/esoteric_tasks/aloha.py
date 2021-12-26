import os

data_links = ['https://cs.uwaterloo.ca/~jhoey/research/aloha/parlai_training_data_cleaned(experimental).zip']
from GenericTools.StayOrganizedTools.download_utils import download_and_unzip







if __name__ == '__main__':
    CDIR = os.path.dirname(os.path.realpath(__file__))
    DATAPATH = os.path.abspath(os.path.join(CDIR, 'data', 'aloha'))

    os.makedirs(DATAPATH, exist_ok=True)
    if len(os.listdir(DATAPATH))==0:
        download_and_unzip(data_links, DATAPATH)
    folder = os.path.join(DATAPATH,os.listdir(DATAPATH)[0])
    print(folder)
    subfolder = os.path.join(folder,os.listdir(folder)[0])
    print(subfolder)
