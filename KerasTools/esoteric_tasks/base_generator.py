import tensorflow as tf

class BaseGenerator(tf.keras.utils.Sequence):

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index=0):
        batch = self.data_generation()
        if self.old:
            return (batch['input_spikes'], batch['mask']), batch['target_output']
        else:
            return (batch['input_spikes'], batch['mask'], batch['target_output']),
