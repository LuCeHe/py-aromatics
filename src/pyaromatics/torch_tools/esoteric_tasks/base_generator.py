
class BaseGenerator:
    def __getitem__(self, index):
        # provide the data in the format [input to model, target output]
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def on_epoch_end(self):
        pass
