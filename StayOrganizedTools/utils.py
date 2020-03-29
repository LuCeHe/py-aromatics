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

import logging
import os
import random
import time
from time import strftime, localtime

import numpy as np
import tensorflow as tf
import yagmail
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
    time_string = time.strftime("%Y-%m-%d-%H-%M-%S", named_tuple)
    return time_string


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
    tf.random.set_seed(seed)
    # tf.random.set_random_seed(seed)

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
        name_experiment='',
        receiver_emails=[]):
    if not isinstance(receiver_emails, list): receiver_emails = [receiver_emails]
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    yag = yagmail.SMTP('my.experiments.336@gmail.com', ':(1234abcd')
    subject = random_string + ' The Experiment is [DONE] ! ' + name_experiment

    logger.info('Sending Results via Email!')
    # send specific files specified
    for filepath in filepaths_list:
        try:
            contents = [filepath]
            for email in receiver_emails:
                yag.send(to=email, contents=contents, subject=subject)
        except:
            pass

    # send content of folders
    for folderpath in folders_list:
        content = os.listdir(folderpath)
        failed = []
        for dir in tqdm(content):
            try:
                path = os.path.join(folderpath, dir)
                contents = [path]
                for email in receiver_emails:
                    yag.send(to=email, contents=contents, subject=subject)
            except:
                failed.append(dir)

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


def simple_email(receiver_emails=[], bcc_receiver_emails=[], text_email=''):
    subject = 'Luce ultravioletta lontana contro il covid19'

    articles = [
        'C:\\Users\\PlasticDiscobolus\\Downloads\\2018 - Far-UVC light A new tool to control the spread of airborne-mediated microbial diseases.pdf',
        'C:\\Users\\PlasticDiscobolus\\Downloads\\2017 - Germicidal Efficacy and Mammalian Skin Safety of 222-nm UV Light.pdf',
        'C:\\Users\\PlasticDiscobolus\\Downloads\\2016 - 207-nm UV Light—A Promising Tool for Safe Low-Cost Reduction of Surgical Site Infections. II In-Vivo Safety Studies.pdf',
        'C:\\Users\\PlasticDiscobolus\\Dropbox\\very serious\\CV Luca\\Luca Celotti - CV.pdf',
    ]
    yag = yagmail.SMTP('luca.herrtti@gmail.com', '')

    contents = [text_email] + articles
    yag.send(to=receiver_emails, bcc=bcc_receiver_emails, contents=contents, subject=subject)


if __name__ == '__main__':
    receiver_emails = ['l.nordio@kedrion.com',
                       'office@angelini.at',
                       'angelinistampa@angelini.it',
                       'info@angelinipharma.com',
                       'press@angelinipharma.com',
                       'business.development@angelini.it ',
                       'networking@angelini.it',
                       'privacydpo@alfasigma.com',
                       'info@alfasigma.es',
                       'info@alfasigma.it',
                       'info.it@alfasigma.com',
                       'groupDPO@recordati.com',
                       'info@bracco.com',
                       'customerservicesgb@bracco.com',
                       'services.professionaleurope@bracco.com',
                       'productservicesgb@bracco.com',
                       'info.at@chiesi',
                       'dpoit@chiesi.com',
                       'diaggare@menarini.it',
                       'luca.herrtti@gmail.com']

    text_email = """
    Gentile signora / signore,

    La mia formazione è in Fisica e Biofisica, ora sto finendo un dottorato in Intelligenza Artificiale presso l'Università di Sherbrooke in Canada.

    Ho trovato un articolo che descrive un metodo fattibile per combattere il covid19: "Luce UVC lontana: un nuovo strumento per controllare la diffusione di malattie microbiche mediate dall'aria". In questo caso, gli autori usano un ultravioletto lontano, con una lunghezza d'onda compresa tra 207-222 nm, e dimostrano la sua efficacia nell'eliminazione dei virus. Hanno anche citato i loro precedenti studi in cui hanno applicato tali radiazioni su un modello animale in vivo e hanno dimostrato che a questa lunghezza d'onda, l'effetto non era statisticamente significativo rispetto al controllo, in cui la luce irradiata non aveva componenti ultravioletti. Pertanto, concludono che detta lunghezza d'onda non ha effetti dannosi sulla pelle del modello animale. Allego tre articoli degli autori.

    Rispetto ad altri lavori, in questo caso sono già stati condotti esperimenti su modelli animali in vivo, quindi è altamente probabile che siano ugualmente sicuri per l'uomo.

    Vorrei collaborare con voi per iniziare a produrre lampade per ospedali, prima una dozzina di prototipi e poi su larga scala, emettendo in modo intermittente a 207 nm, ogni 12 ore, un'esposizione di 6 ore. L'idea sarebbe quella di sostituire le lampade che sono già installate nei soffitti per lampade disinfettanti. Vorrei anche iniziare a produrre lampade a mano per disinfettare luoghi più nascosti e installarli su robot tipo roomba, per automatizzare il processo e liberare il maggior numero possibile di personale sanitario dalle attività di disinfezione.

    Ho anche sentito che in alcuni ospedali hanno chiesto donazioni di sacchetti di plastica per telefoni cellulari, perché erano soliti metterli tutti insieme, aumentando la probabilità di contagio. Vorrei quindi collaborare con voi per fornire agli ospedali stazioni in cui i telefoni cellulari possano essere lasciati e irradiarli a 207 nm.

    Grazie mille in anticipo,
    Luca Celotti Herranz"""

    simple_email(receiver_emails=[], bcc_receiver_emails=receiver_emails, text_email=text_email)
