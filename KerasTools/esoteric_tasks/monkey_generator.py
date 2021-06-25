import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from GenericTools.KerasTools.esoteric_tasks.base_generator import BaseGenerator

events = [
    'TS-ON', 'TS-OFF', 'WS-ON', 'WS-OFF', 'PG-ON', 'SG-ON',
    'CUE-OFF', 'LF-ON', 'HF-ON', 'SR', 'HS', 'RW-ON', 'GO-OFF/RW-OFF',
    'STOP', 'ERROR'
]

events_idx = {k: v for v, k in enumerate(events)}
output_type = {
    'PG-ONHF-ON': 0,
    'SG-ONHF-ON': 1,
    'PG-ONLF-ON': 2,
    'SG-ONLF-ON': 3,
}


def trial_structure(grip_type, force_type, n_repeat):
    random_decision_time = 3
    random_beginning = 3
    random_beginning = np.random.choice(
        range(int(random_beginning * n_repeat / 3), random_beginning * n_repeat))
    random_decision_time = np.random.choice(
        range(int(random_decision_time * n_repeat / 3), random_decision_time * n_repeat))
    return [events_idx['WS-OFF']] * random_beginning + \
           [events_idx['TS-ON']] * 4 * n_repeat + [events_idx['WS-ON']] * 4 * n_repeat + \
           [events_idx[grip_type]] * 4 * n_repeat + [events_idx['CUE-OFF']] * 10 * n_repeat \
           + [events_idx[force_type]] * random_decision_time \
           + [events_idx['SR']] * 3 * n_repeat + [events_idx['HS']] * 5 * n_repeat + [
               events_idx['RW-ON']] * 3 * n_repeat \
           + [events_idx['WS-OFF']] * 2 * n_repeat


class MonkeyGenerator(BaseGenerator):
    def __init__(
            self,
            batch_size=2,
            steps_per_epoch=2,
            epochs=1,
            **kwargs):
        super().__init__(**kwargs)

        self.n_repeat = 100

        max_len = (3 + 4 + 4 + 4 + 10 + 3 + 3 + 5 + 3 + 2) * self.n_repeat
        self.batch_size = batch_size

        self.in_dim = 1
        self.out_dim = len(events_idx)
        self.in_len = max_len
        self.out_len = max_len

        self.epochs = 30 if epochs == None else epochs

        self.steps_per_epoch = 10 if steps_per_epoch == None else steps_per_epoch

    def data_generation(self):
        inputs = []
        outputs = []
        masks = []
        for _ in range(self.batch_size):
            grip_type = np.random.choice(['PG-ON', 'SG-ON'])
            force_type = np.random.choice(['HF-ON', 'LF-ON'])

            trial_input = trial_structure(grip_type, force_type, self.n_repeat)
            trial_input = np.array(trial_input)

            trial_mask = trial_input == events_idx['HS']
            trial_output = trial_mask * output_type[grip_type + force_type]

            inputs.append(trial_input)
            masks.append(trial_mask)
            outputs.append(trial_output)

        padded_in = pad_sequences(inputs, maxlen=self.in_len, value=events_idx['WS-OFF'])[..., None]
        padded_out = pad_sequences(outputs, maxlen=self.out_len, value=0)
        mask = pad_sequences(masks * 1, maxlen=self.out_len, value=0)[..., None]

        oh_out = tf.keras.utils.to_categorical(padded_out, num_classes=self.out_dim)
        mask = np.repeat(mask, self.out_dim, axis=-1)
        oh_out = oh_out*mask

        return {'input_spikes': padded_in, 'target_output': oh_out, 'mask': mask}


if __name__ == '__main__':
    print(events_idx)
    grip_type = 'PG-ON'  # 'PG-ON' 'SG-ON'
    force_type = 'HF-ON'  # 'HF-ON' 'LF-ON'
    n_repeat = 1  # 100

    trial_input = trial_structure(grip_type, force_type, n_repeat)
    trial_input = np.array(trial_input)

    trial_mask = trial_input == events_idx['HS']
    trial_output = trial_mask * output_type[grip_type + force_type]
    print(trial_input)
    print(trial_mask)
    print('Trials have a max length of {}ms'.format(len(trial_input) * 100))
    print(output_type[grip_type + force_type])
    print(trial_output)

    batch_size = 3
    generator = MonkeyGenerator(batch_size=batch_size)

    batch = generator.data_generation()
    print(batch['input_spikes'].shape)
    print(batch['target_output'].shape)
    print(batch['mask'].shape)
    print(batch['input_spikes'])
    print(batch['target_output'])
