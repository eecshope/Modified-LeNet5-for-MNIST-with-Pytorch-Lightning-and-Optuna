from source.fashion_mnist_cnn import LeNet5
import argparse
import os
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import _pickle as pkl
import optuna
from optuna.integration import PyTorchLightningPruningCallback


class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


parser = argparse.ArgumentParser(description="Grid Search for the Best Hyper parameters")
parser.add_argument(
    "--pruning",
    "-p",
    action="store_true",
    help="Activate the pruning feature. `MedianPruner` stops unpromising "
    "trials at the early stages of training.",
)
parser.add_argument("--model_dir")
parser.add_argument("--data_dir")
parser.add_argument("--gpu")
args = parser.parse_args()

# get ready for the components for training
MODEL_DIR = args.model_dir
pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

# load the data and build the dataloader
data_train = FashionMNIST(args.data_dir,
                          download=False,
                          transform=transforms.Compose([
                              transforms.Resize((32, 32)),
                              transforms.ToTensor()]))

data_val = FashionMNIST(args.data_dir,
                        train=False,
                        download=False,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor()]))

data_val.data = data_val.data[:1000]
data_val.targets = data_val.targets[:1000]


# define the optimizing process
def objective(trial: optuna.Trial):
    # Filenames for each trial must be made unique in order to access each checkpoint.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"), monitor="val_acc"
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We don't use any logger here as it requires us to implement several abstract
    # methods. Instead we setup a simple callback, that saves metrics from each validation step.
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=checkpoint_callback,
        max_epochs=50,
        gpus=args.gpu if torch.cuda.is_available() else None,
        callbacks=[metrics_callback],
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_acc")
    )

    model = LeNet5(trial)
    bsz = trial.suggest_int("bsz", 32, 128, 32)
    train_loader = DataLoader(data_train, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=1)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    return metrics_callback.metrics[-1]["val_acc"]


# start to find out
study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
best_trial = study.best_trial

print("  Value: {}".format(best_trial.value))

print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

with open('study.pkl', 'wb') as file:
    pkl.dump(study, file)
