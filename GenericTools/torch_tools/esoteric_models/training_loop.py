import torch, math, tqdm

BASELINE_MODELS = [
    "ncde",
    "odernn",
    "dt",
    "decay",
    "gruode",
    "odernn_forecasting",
    "ncde_forecasting",
    "decay_forecasting",
    "dt_forecasting",
    "gruode_forecasting",
    "double_ncde_new6",
]
BASELINE_MODELS_F = [
    "ncde",
    "odernn",
    "dt",
    "decay",
    "gruode",
    "odernn_forecasting",
    "decay_forecasting",
    "dt_forecasting",
]
NCDE_BASELINES = ["gruode_forecasting", "ncde_forecasting"]


def update_kwargs(kwargs, model_name, slope):
    if model_name in NCDE_BASELINES:
        kwargs.update(stream=True)
    elif not model_name in BASELINE_MODELS_F:
        kwargs.update(stream=True, slope=slope)
        # pred_y = model(times, train_coeffs, lengths, slope, stream=True, **kwargs)
    return kwargs


class _SuppressAssertions:
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write("Caught AssertionError: " + str(exc_val))
            return True


def _evaluate_metrics(model_name, generator, model, metric, slope, device, kwargs):
    with torch.no_grad():
        total_dataset_size = 0
        total_loss = 0
        steps_per_epoch = len(generator)
        print(steps_per_epoch)

        batch_size = generator.batch_size
        for i in range(steps_per_epoch):
            print(i)
            batch = generator.__getitem__(i)
            batch = tuple(b.to(device) for b in batch)

            if len(batch) == 7:
                *coeffs, true_y, lengths, times = batch
            elif len(batch) == 2:
                coeffs, true_y = batch
                times, lengths = None, None

            kwargs = update_kwargs(kwargs, model_name, slope)
            pred_y = model(times, coeffs, lengths, **kwargs)

            total_dataset_size += batch_size

            total_loss += metric(pred_y, true_y) * batch_size

        total_loss /= total_dataset_size
        return total_loss.item()


class ModelWrapper():
    def __init__(self, model, device, model_name='', slope_check=False):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.slope_check = slope_check

    def compile(self, loss, optimizer, metrics=[]):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics + [loss]

    def fit(self, train_generator, val_generator=None, test_generator=None):
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        best_train_loss = math.inf
        best_val_loss = math.inf

        generators = {'train': train_generator, 'val': val_generator, 'test': test_generator}
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        epochs = train_generator.epochs
        steps_per_epoch = len(train_generator)
        tqdm_range = tqdm.tqdm(range(epochs))
        history = []
        breaking = False

        for epoch in tqdm_range:

            if self.slope_check:
                slope = (epoch * 0.12) + 1.0
            else:
                slope = 0.0
            kwargs = update_kwargs({}, self.model_name, slope)

            if breaking:
                break

            for i in range(steps_per_epoch):
                batch = train_generator.__getitem__(i)
                batch = tuple(b.to(self.device) for b in batch)
                if breaking:
                    break
                with _SuppressAssertions(tqdm_range):

                    if len(batch) == 7:
                        *train_coeffs, true_y, lengths, times = batch
                        true_y = true_y.float()
                    elif len(batch) == 2:
                        train_coeffs, true_y = batch
                        times, lengths = None, None

                    pred_y = self.model(times, train_coeffs, lengths, **kwargs)

                    loss = self.loss(pred_y, true_y)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            train_generator.on_epoch_end()

            if epoch % epoch_per_metric == 0 or epoch == epochs - 1:
                self.model.eval()
                epoch_metrics = {}
                for k, generator in generators.items():
                    print(k)
                    if not generator is None:
                        metrics_evaluate = {}
                        for metric in self.metrics:
                            print(metric.__name__)
                            metric_value = _evaluate_metrics(self.model_name, generator, self.model, metric, slope,
                                                             self.device, kwargs)
                            metrics_evaluate[metric.__name__] = metric_value
                        epoch_metrics[k] = metrics_evaluate
                        generator.on_epoch_end()
                self.model.train()

                train_loss = list(epoch_metrics['train'].values())[-1]
                if train_loss * 1.0001 < best_train_loss:
                    best_train_loss = train_loss
                    best_train_loss_epoch = epoch

                val_loss = list(epoch_metrics['val'].values())[-1]
                if val_loss * 1.0001 < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_epoch = epoch

                performance_string = ''
                for set, metrics in epoch_metrics.items():
                    performance_string += set + ': '
                    for n, m in metrics.items():
                        performance_string += '{}: {}, '.format(n, m)
                tqdm_range.write(
                    "Epoch: {} {} slope: {:.5}".format(epoch, performance_string, slope)
                )

                scheduler.step(val_loss)
                epoch_metrics.update(epoch=epoch)
                history.append(epoch_metrics)

                if epoch > best_train_loss_epoch + plateau_terminate:
                    tqdm_range.write(
                        "Breaking because of no improvement in training loss for {} epochs."
                        "".format(plateau_terminate)
                    )
                    breaking = True

        return history
