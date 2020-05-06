from source.fashion_mnist_cnn import LeNet5
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description="Test LeNet5")
parser.add_argument('--ckpt_path')
parser.add_argument('--data_dir')
parser.add_argument('--gpu')
args = parser.parse_args()

classifier = LeNet5.load_from_checkpoint(args.ckpt_path, None, None, 5, 32, 0)

test_data = FashionMNIST(args.data_dir,
                         train=False,
                         download=False,
                         transform=transforms.Compose([
                             transforms.Resize((32, 32)),
                             transforms.ToTensor()]))
test_data.data = test_data.data[1000:]
test_data.targets = test_data.targets[1000:]

test_loader = DataLoader(test_data, batch_size=8)

trainer = Trainer(gpus=args.gpu)
trainer.test(classifier, test_dataloaders=test_loader)
