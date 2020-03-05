import logging
import os
import pathlib
from time import strftime, localtime

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


class CustomFileStorageObserver(FileStorageObserver):

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        if _id is None:
            # create your wanted log dir
            time_string = strftime("%Y-%m-%d-at-%H-%M-%S", localtime())
            timestamp = "experiment-{}________".format(time_string)
            options = '_'.join(meta_info['options']['UPDATE'])
            run_id = timestamp + options

            # update the basedir of the observer
            self.basedir = os.path.join(self.basedir, run_id)
            self.basedir = os.path.join(ex_info['base_dir'], self.basedir)

            # and again create the basedir
            pathlib.Path(self.basedir).mkdir(exist_ok=True, parents=True)

        return super().started_event(ex_info, command, host_info, start_time, config, meta_info, _id)


def CustomExperiment(experiment_name, base_dir=None):
    ex = Experiment(name=experiment_name, base_dir=base_dir)
    #ex.observers.append(FileStorageObserver.create("experiments"))
    ex.observers.append(CustomFileStorageObserver.create("experiments"))

    # ex.observers.append(MongoObserver())
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    # set up a custom logger
    logger = logging.getLogger('mylogger')
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname).1s] %(name)s >> "%(message)s"')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')

    # attach it to the experiment
    ex.logger = logger

    data_path = os.path.join(base_dir, 'data')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    return ex
