# -*- coding: UTF-8 -*-

import pytorch_lightning as pl
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict
import argparse
from typing import Union


# LeNet-5
class LeNet5(pl.LightningModule):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    F7 - 10 (Output)

    Feel free to try different filter numbers
    """

    def __init__(self, hparams: Union[argparse.Namespace, optuna.Trial]):
        super(LeNet5, self).__init__()

        if isinstance(hparams, argparse.Namespace):
            kernel_size = hparams.kernel_size
            kernel_num = hparams.kernel_num
            got_three = hparams.got_three
        else:
            kernel_size = hparams.suggest_int('kernel_size', 3, 5, 7)
            kernel_num = hparams.suggest_int('kernel_num', 16, 32, 64)
            got_three = hparams.suggest_int('got_three', 0, 1)
        pad_len = (kernel_size - 1) // 2
        # edge_length = 32
        if got_three:
            seq = OrderedDict([
                ('c1', nn.Conv2d(1, 6, kernel_size=(kernel_size, kernel_size), padding=pad_len)),
                ('relu1', nn.ReLU()),
                ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                ('c3', nn.Conv2d(6, kernel_num, kernel_size=(kernel_size, kernel_size), padding=pad_len)),
                ('relu3', nn.ReLU()),
                ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                ('c5', nn.Conv2d(kernel_num, 120, kernel_size=(kernel_size, kernel_size), padding=pad_len)),
                ('relu5', nn.ReLU())
            ])
            input_length = 120 * 8 * 8
        else:
            seq = OrderedDict([
                ('c1', nn.Conv2d(1, 6, kernel_size=(kernel_size, kernel_size), padding=pad_len)),
                ('relu1', nn.ReLU()),
                ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                ('c3', nn.Conv2d(6, kernel_num, kernel_size=(kernel_size, kernel_size), padding=pad_len)),
                ('relu3', nn.ReLU()),
            ])
            input_length = kernel_num * 16 * 16
        self.convnet = nn.Sequential(seq)

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(input_length, 84)),
            ('relu6', nn.ReLU()),
            ('dropout', nn.Dropout()),
            ('f7', nn.Linear(84, 10))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

    def training_step(self, batch, batch_index):
        image, label = batch
        logits = self(image)
        loss = f.cross_entropy(logits, label)
        log = {"training_loss": loss.detach()}
        progress_bar = log
        output = {"loss": loss,
                  "progress_bar": progress_bar,
                  "log": log}
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler}

    def validation_step(self, batch, batch_idx):
        image, label = batch
        with torch.no_grad():
            logits = self(image)
            loss = f.cross_entropy(logits, label, reduction='sum')
            preds = torch.argmax(logits, dim=1)
            acc = (preds == label).sum().to(torch.float32)
        output = {"loss": loss, "acc": acc, "num": label.shape[0]}
        return output

    def validation_epoch_end(self, outputs):
        loss = 0
        acc = 0
        num = 0
        for output in outputs:
            loss += output["loss"]
            acc += output["acc"]
            num += output["num"]
        avg_loss = loss / num
        avg_acc = acc / num
        log = {"val_loss": avg_loss, "val_acc": avg_acc}
        progress_bar = {"val_acc": avg_acc}
        output = {"log": log,
                  "progress_bar": progress_bar}
        return output

    def test_step(self, batch, batch_idx):
        image, label = batch
        with torch.no_grad():
            logits = self(image)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == label).sum().to(torch.float32)
            output = {"acc": acc, "num": label.shape[0]}
        return output

    def test_epoch_end(
            self,
            outputs
    ):
        num = 0
        acc = 0
        for output in outputs:
            acc += output['acc']
            num += output['num']
        test_acc = acc / num
        progress_bar = {"test_acc": test_acc}
        log = progress_bar
        return {"log": log, "progress_bar": progress_bar}
