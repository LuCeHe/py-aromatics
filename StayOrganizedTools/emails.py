import os, shutil, yagmail, logging
from tqdm import tqdm
import numpy as np

logger = logging.getLogger('mylogger')

def email_results(
        folders_list=[],
        filepaths_list=[],
        text='',
        name_experiment='',
        except_files=None,
        receiver_emails=[]):
    if not isinstance(receiver_emails, list): receiver_emails = [receiver_emails]
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    yag = yagmail.SMTP('my.experiments.336@gmail.com', ':(1234abcd')
    subject = random_string + ' The Experiment is [DONE] ! ' + name_experiment

    logger.info('Sending Results via Email!')
    # send specific files specified
    for filepath in filepaths_list + [text]:
        try:
            contents = [filepath]
            for email in receiver_emails:
                yag.send(to=email, contents=contents, subject=subject)
        except Exception as e:
            print(e)

    # send content of folders
    for folderpath in folders_list:
        content = os.listdir(folderpath)
        failed = []
        for file in tqdm(content):
            if except_files == None or not except_files in file:
                try:
                    path = os.path.join(folderpath, file)
                    contents = [path]
                    for email in receiver_emails:
                        yag.send(to=email, contents=contents, subject=subject)
                except Exception as e:
                    failed.append(file)
                    print(e)

        contents = ['among all the files\n\n{} \n\nthese failed to be sent: \n\n{}'.format('\n'.join(content),
                                                                                           '\n'.join(failed))]
        for email in receiver_emails:
            yag.send(to=email, contents=contents, subject=subject)


def email_folder_content(folderpath, receiver_email=''):
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    subject = random_string + ' The Experiment is [DONE] !'

    content = os.listdir(folderpath)
    print('content of the folder:\n')
    for dir in content:
        print('  ', dir)

    if input("\n\nare you sure? (y/n)") != "y":
        exit()

    yag = yagmail.SMTP('my.experiments.336@gmail.com', ':(1234abcd')
    failed = []
    for dir in tqdm(content):
        try:
            path = os.path.join(folderpath, dir)
            contents = [path]
            yag.send(to=receiver_email, contents=contents, subject=subject)
        except:
            failed.append(dir)

    contents = ['among all the files\n\n{} \n\nthese failed to be sent: \n\n{}'.format('\n'.join(content),
                                                                                       '\n'.join(failed))]
    yag.send(to=receiver_email, contents=contents, subject=subject)



def CompressAndSend(path_folders, email):
    ds = os.listdir(path_folders)
    paths = [os.path.join(path_folders, d) for d in ds]
    for d, path in zip(ds, paths):
        shutil.make_archive(d, 'zip', path)

    ds = [d for d in os.listdir(path_folders) if '.zip' in d]
    email_results(
        filepaths_list=ds,
        name_experiment=' compressed folders ',
        receiver_emails=[email])


def SendFilesWithIdentifier(container_dir, email_to, files_identifier):
    ds = [d for d in os.listdir(container_dir) if files_identifier in d]

    email_results(
        filepaths_list=ds,
        name_experiment=' identified files ',
        receiver_emails=[email_to])
