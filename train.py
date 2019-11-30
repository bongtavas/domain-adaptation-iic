from torchsummary import summary
from models.lenetplus import LeNetPlus
from IID_losses import IID_loss
from dataset import MNIST_SVHN
from torch.utils.data import DataLoader
from utils import get_opt, update_lr
from eval_metrics import hungarian_match, accuracy
import numpy as np
import argparse
import time
import logging
import torch
import os

np.random.seed(5241)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_dest_samples", type=int, default=10)
    parser.add_argument("--opt", type=str, default="Adam")
    parser.add_argument("--mode", type=str, default="IID")
    parser.add_argument("--lamb", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
    parser.add_argument("--lr_mult", type=float, default=0.1)

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_sz", type=int, default=32)  # num pairs
    parser.add_argument("--results_path", type=str, default="results")
    parser.add_argument("--save_freq", type=int, default=10)

    config = parser.parse_args()

    return config


def train(config):
    dataset = MNIST_SVHN(K=10)

    train_loader = DataLoader(dataset, batch_size=config.batch_sz, shuffle=True)

    net_model = LeNetPlus()

    net_model.cuda()

    optimiser = get_opt(config.opt)(net_model.parameters(), lr=config.lr)

    best_acc = 0.

    model_save_path = os.path.join(config.results_path, "iic-adapt.pt")

    for epoch in range(1, config.num_epochs + 1):
        start = time.time()
        epoch_loss = 0.
        n = 0

        if epoch in config.lr_schedule:
            optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

        for idx, batch in enumerate(train_loader):
            mnist_x, svhn_x, label_y = batch
            if torch.cuda.is_available():
                mnist_x = mnist_x.cuda()
                svhn_x = svhn_x.cuda()
                label_y = label_y.cuda()

            z = net_model(mnist_x)
            z_prime = net_model(svhn_x)

            loss, loss_no_lamb = IID_loss(z, z_prime)
            epoch_loss += loss.item()
            n += 1

            loss.backward()
            optimiser.step()

        end = time.time()
        logging.info("=== Epoch {%s}   Loss: {%.4f}  Running time: {%4f}" % (str(epoch), (epoch_loss) / n, end - start))

        acc = validate(net_model, dataset, config)

        logging.info("Validation Acc: {%.4f}" % (acc))

        if acc > best_acc:
            logging.info("Validation Acc improved, saving model")
            torch.save(net_model.state_dict, model_save_path)


def validate(net_model, dataset, config):
    net_model.eval()
    val_loader = DataLoader(dataset, batch_size=config.batch_sz, shuffle=False)

    preds, targets = get_preds_and_targets(net_model, val_loader, dataset)

    matches = hungarian_match(preds, targets, num_classes=config.num_classes)

    new_preds = get_preds_actual_class(preds, matches, config)

    acc = accuracy(new_preds, targets, config.num_classes)

    net_model.train()

    return acc


def get_preds_and_targets(net_model, val_loader, dataset):
    dataset_size = len(dataset) + config.num_dest_samples
    K = config.num_dest_samples
    preds = torch.zeros(dataset_size + (K * config.num_classes))
    targets = torch.zeros(dataset_size + (K * config.num_classes))
    batch_sz = config.batch_sz
    i = 0
    # get predictions for the src dataset
    for b_i, batch in enumerate(val_loader):
        src_x, _, label = batch
        if torch.cuda.is_available():
            src_x = src_x.cuda()
        with torch.no_grad():
            src_y = net_model(src_x)  # (batch_size, num_classes)
        preds[i:i + batch_sz] = torch.argmax(src_y, dim=1)
        targets[i:i + batch_sz] = label

        i += batch_sz

    # get predictions from the few samples of dest dataset
    j = len(dataset)
    for label in dataset.svhn_class_indices:
        dest_x_indices = dataset.svhn_class_indices[label]
        dest_x = dataset.svhn_x[dest_x_indices]
        if torch.cuda.is_available():
            dest_x = dest_x.cuda()
        with torch.no_grad():
            dest_y = net_model(dest_x)
        preds[j:j + K] = torch.argmax(dest_y, dim=1)
        targets[j:j + K] = dataset.svhn_y[dest_x_indices]
        j += K

    if torch.cuda.is_available():
        preds = preds.cuda()
        targets = targets.cuda()

    return preds, targets


def get_preds_actual_class(preds, matches, config):
    num_samples = preds.size(0)
    new_preds = torch.zeros(num_samples, dtype=preds.dtype)
    found = torch.zeros(config.num_classes)

    if torch.cuda.is_available():
        new_preds = new_preds.cuda()

    for pred_i, target_i in matches:
        new_preds[torch.eq(preds, int(pred_i))] = torch.from_numpy(np.array(target_i)).cuda().int().item()
        found[pred_i] = 1

    assert (found.sum() == config.num_classes)  # each output_k must get mapped

    return new_preds


if __name__ == "__main__":

    config = parse_args()

    train(config)
