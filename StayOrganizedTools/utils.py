# Copyright (c) 2018, 
#
# authors: Luca Celotti
# during their PhD at Universite' de Sherbrooke
# under the supervision of professor Jean Rouat
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import argparse, shutil, logging, os, random, time, yagmail
from time import strftime, localtime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger('mylogger')


def make_directories(time_string=None):
    experiments_folder = "experiments"
    if not os.path.isdir(experiments_folder):
        os.mkdir(experiments_folder)

    if time_string == None:
        time_string = strftime("%Y-%m-%d-at-%H:%M:%S", localtime())

    experiment_folder = experiments_folder + '/experiment-' + time_string + '/'

    if not os.path.isdir(experiment_folder):
        os.mkdir(experiment_folder)

        # create folder to save new models trained
    model_folder = experiment_folder + '/model/'
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

        # create folder to save TensorBoard
    log_folder = experiment_folder + '/log/'
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)

    return experiment_folder


def plot_softmax_evolution(softmaxes_list, name='softmaxes'):
    import matplotlib.pylab as plt

    f = plt.figure()
    index = range(len(softmaxes_list[0]))
    for softmax in softmaxes_list:
        plt.bar(index, softmax)

    plt.xlabel('Token')
    plt.ylabel('Probability')
    plt.title('softmax evoluti\on during training')
    plt.show()
    f.savefig(name + ".pdf", bbox_inches='tight')


def checkDuringTraining(generator_class, indices_sentences, encoder_model, decoder_model, batch_size, lat_dim):
    # original sentences
    sentences = generator_class.indicesToSentences(indices_sentences)

    # reconstructed sentences
    point = encoder_model.predict(indices_sentences)
    indices_reconstructed, _ = decoder_model.predict(point)

    sentences_reconstructed = generator_class.indicesToSentences(indices_reconstructed)

    # generated sentences
    noise = np.random.rand(batch_size, lat_dim)
    indicess, softmaxes = decoder_model.predict(noise)
    sentences_generated = generator_class.indicesToSentences(indicess)

    from prettytable import PrettyTable

    table = PrettyTable(['original', 'reconstructed', 'generated'])
    for b, a, g in zip(sentences, sentences_reconstructed, sentences_generated):
        table.add_row([b, a, g])
    for column in table.field_names:
        table.align[column] = "l"
    print(table)

    print('')
    print('number unique generated sentences:   ', len(set(sentences_generated)))
    print('')
    print(softmaxes[0][0])
    print('')

    return softmaxes


def timeStructured():
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    return time_string + random_string


def collect_information():
    f = open('./cout.txt', 'r')
    i = 0

    for line in f:
        if '>> "           AriEL' in line:
            print(i)
            i = 0
            print(line)
            g = open(line[-11:-2] + '.txt', 'w')
            g.write('\n\n')
            g.write(line)
            g.write('\n\n')

        if 'cpu' in line or 'CPU' in line:
            if not 'I tensorflow' in line:
                i += 1
                g.write(line)
                g.write('\n')
    print(i)


def setReproducible(seed=0, disableGpuMemPrealloc=True):
    # Fix the seed of all random number generator
    random.seed(seed)
    np.random.seed(seed)
    if tf.__version__[0] == '2':
        tf.random.set_seed(seed)
    else:
        tf.random.set_random_seed(seed)

    config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        device_count={'CPU': 1})
    if disableGpuMemPrealloc:
        config.gpu_options.allow_growth = True
    # K.clear_session()
    # K.set_session(tf.Session(config=config))


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


def Dict2ArgsParser(args_dict):
    parser = argparse.ArgumentParser()
    for k, v in args_dict.items():
        if type(v) == bool and v == True:
            parser.add_argument("--" + k, action="store_true")
        elif type(v) == bool and v == False:
            parser.add_argument("--" + k, action="store_false")
        else:
            parser.add_argument("--" + k, type=type(v), default=v)
    args = parser.parse_args()
    return args


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--container_dir", type=str, default='', help="Where are the files of interest located.")
    parser.add_argument("--email_to", type=str, default='', help="Who should receive them.")
    parser.add_argument("--files_identifier", type=str, default=".zip", help="Which files are you interested in.")
    args = parser.parse_args()

    SendFilesWithIdentifier(container_dir=args.container_dir, email_to=args.email_to,
                            files_identifier=args.files_identifier)
