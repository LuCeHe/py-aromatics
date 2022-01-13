import numpy as np
import tensorflow as tf
from GenericTools.StayOrganizedTools.utils import str2val

split_names = ['valid_random_split', 'valid_topic_split', 'test_random_split', 'test_topic_split', 'train']


# split_names = ['train', 'valid_random_split', 'test_random_split']


class RandomTask(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            epochs=1,
            batch_size=32,
            steps_per_epoch=1,
            string_config=None,
    ):
        self.__dict__.update(
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
        )

        self.vocabsize = str2val(string_config, 'vocabsize', int, default=32)

        self.input_types = str2val(string_config, 'intypes', str, default='int').replace('[', '').replace(']', '').split(',')
        input_shapes = str2val(string_config, 'inshapes', str, default='(32,)') \
            .replace('[', '').replace(']', '').split('),(')
        input_shapes = [s.replace('(', '').replace(')', '') for s in input_shapes]
        self.input_shapes = [tuple([int(i) for i in s.split(',') if not i is '']) for s in input_shapes]

        self.output_types = str2val(string_config, 'outtypes', str, default='int') \
            .replace('[', '').replace(']', '').split(',')
        output_shapes = str2val(string_config, 'outshapes', str, default='(32,)') \
            .replace('[', '').replace(']', '').split('),(')
        output_shapes = [s.replace('(', '').replace(')', '') for s in output_shapes]
        self.output_shapes = [tuple([int(i) for i in s.split(',') if not i is '']) for s in output_shapes]

        self.epochs = 50 if epochs == None else epochs
        self.steps_per_epoch = 3 if steps_per_epoch == None else steps_per_epoch

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        in_batches = [
            random_batch(intype, (self.batch_size, *inshape), self.vocabsize if intype == 'int' else 1)
            for inshape, intype in zip(self.input_shapes, self.input_types)
        ]

        out_batches = [
            random_batch(outtype, (self.batch_size, *outshape), self.vocabsize if outtype == 'int' else 1)
            for outshape, outtype in zip(self.input_shapes, self.input_types)
        ]
        return in_batches, out_batches


def random_batch(batch_type, batch_shape, max_value):
    if batch_type == 'int':
        batch = np.random.choice(max_value, size=batch_shape)

    elif batch_type == 'float':
        batch = np.random.uniform(low=-max_value, high=max_value, size=batch_shape)

    else:
        raise NotImplementedError

    return batch


if __name__ == '__main__':
    string_config = 'random_intypes:[int,int,float]_inshapes:[(3,4),(3,),(3,)]' \
                    '_outtypes:[int]_outshapes:[(3,)]_vocabsize:128'
    # gen = RandomTask(string_config)

    vocabsize = str2val(string_config, 'vocabsize', int, default=32)
    batch_size = 2

    input_types = str2val(string_config, 'intypes', str, default='int').replace('[', '').replace(']', '').split(',')
    input_shapes = str2val(string_config, 'inshapes', str, default='(32,)').replace('[', '').replace(']', '').split(
        '),(')
    input_shapes = [s.replace('(', '').replace(')', '') for s in input_shapes]
    input_shapes = [tuple([int(i) for i in s.split(',') if not i is '']) for s in input_shapes]

    output_types = str2val(string_config, 'outtypes', str, default='int').replace('[', '').replace(']', '').split(',')
    output_shapes = str2val(string_config, 'outshapes', str, default='(32,)').replace('[', '').replace(']', '').split(
        '),(')
    output_shapes = [s.replace('(', '').replace(')', '') for s in output_shapes]
    output_shapes = [tuple([int(i) for i in s.split(',') if not i is '']) for s in output_shapes]

    in_batches = [
        random_batch(intype, (batch_size, *inshape), vocabsize if intype == 'int' else 1)
        for inshape, intype in zip(input_shapes, input_types)
    ]

    out_batches = [
        random_batch(outtype, (batch_size, *outshape), vocabsize if outtype == 'int' else 1)
        for outshape, outtype in zip(input_shapes, input_types)
    ]
