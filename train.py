from models.net5g import ClusterNet5g
from dataset import MNIST_SVHN
from torch.utils.data import DataLoader
from utils.IID_losses import IID_loss
from utils.train_utils import get_opt, update_lr
from utils.eval_metrics import hungarian_match, accuracy
import numpy as np
import argparse
import time
import logging
import torch
import os
from torchvision import datasets
from torchvision import transforms
from utils.transforms import sobel_process, custom_greyscale_to_tensor

np.random.seed(5241)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def parse_config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_dest_per_class", type=int, default=10)
    parser.add_argument("--num_sub_heads", type=int, default=5)
    parser.add_argument("--batchnorm_track", default=False, action="store_true")
    parser.add_argument("--opt", type=str, default="Adam")
    parser.add_argument("--mode", type=str, default="IID")
    parser.add_argument("--lamb", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
    parser.add_argument("--lr_mult", type=float, default=0.1)

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_sz", type=int, default=700)  # num pairs
    parser.add_argument("--results_path", type=str, default="results")
    parser.add_argument("--save_freq", type=int, default=10)

    parser.add_argument("--output_k_A", type=int, default=50)
    parser.add_argument("--output_k_B", type=int, default=10)

    parser.add_argument("--input_sz", type=int, default=32)
    parser.add_argument("--rand_crop_sz", type=int, default=20)
    parser.add_argument("--include_rgb", default=False, action="store_true")

    config = parser.parse_args()

    config.output_k = config.output_k_B

    if config.include_rgb:
        config.in_channels = 5
    else:
        config.in_channels = 2

    return config


def train(config):
    dataset = MNIST_SVHN(config)

    train_loader = DataLoader(dataset, batch_size=config.batch_sz, shuffle=True)

    net_model = ClusterNet5g(config)

    net_model.cuda()

    net_model = torch.nn.DataParallel(net_model)

    optimiser = get_opt(config.opt)(net_model.module.parameters(), lr=config.lr)

    best_acc = 0.

    model_save_path = os.path.join(config.results_path, "iic-adapt.pt")

    for epoch in range(1, config.num_epochs + 1):
        start = time.time()

        # if epoch in config.lr_schedule:
        #     optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

        for batch_i, batch in enumerate(train_loader):
            net_model.module.zero_grad()
            mnist_x, svhn_x, label_y = batch
            mnist_x = mnist_x.cuda()
            svhn_x = svhn_x.cuda()
            label_y = label_y.cuda()

            mnist_x = sobel_process(mnist_x, config.include_rgb)
            svhn_x = sobel_process(svhn_x, config.include_rgb)

            z = net_model(mnist_x)
            z_prime = net_model(svhn_x)

            avg_loss_batch = None  # avg over the heads
            avg_loss_no_lamb_batch = None
            for i in range(config.num_sub_heads):
                loss, loss_no_lamb = IID_loss(z[i], z_prime[i], config.lamb)
                if avg_loss_batch is None:
                    avg_loss_batch = loss
                    avg_loss_no_lamb_batch = loss_no_lamb
                else:
                    avg_loss_batch += loss
                    avg_loss_no_lamb_batch += loss_no_lamb

            avg_loss_batch /= config.num_sub_heads
            avg_loss_no_lamb_batch /= config.num_sub_heads
            end = time.time()

            if batch_i % 40 == 0:
                logging.info("=== Epoch {%s} Batch: {%d} Avg Loss: {%f} Avg Loss No Lamb: {%f}  Running time: {%4f}" % (str(epoch), batch_i, avg_loss_batch.item(), avg_loss_no_lamb_batch.item(), end - start))

            avg_loss_batch.backward()
            optimiser.step()

        if epoch % 1 == 0:
            stats = evaluate(net_model, config)
            logging.info(str(stats))
            if stats["best"] > best_acc:
                best_acc = stats["best"]
                logging.info("Accuracy improved, saving model")
                torch.save(net_model.state_dict, model_save_path)


def evaluate(net_model, config):
    net_model.eval()
    dest_stats = eval_dest(net_model)
    net_model.train()

    return dest_stats


def eval_dest(net_model):
    dest_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transforms.Compose([
        transforms.CenterCrop(config.rand_crop_sz),
        transforms.Resize(config.input_sz),
        custom_greyscale_to_tensor(config.include_rgb),
    ]))
    dest_loader = DataLoader(dest_dataset, batch_size=config.batch_sz, shuffle=False)

    preds, targets = get_preds_and_targets(config, net_model, dest_loader, dest_dataset)

    matches = []
    accs = []
    for i in range(config.num_sub_heads):
        match = hungarian_match(preds[i], targets, num_classes=config.num_classes)
        matches.append(match)
        actual_preds = get_preds_actual_class(preds[i], match, config)
        acc = accuracy(actual_preds, targets, config.num_classes)
        accs.append(acc)

    best_subhead = np.argmax(accs)
    worst_subhead = np.argmin(accs)
    return {
        "accs": accs,
        "avg": np.mean(accs),
        "std": np.std(accs),
        "best": accs[best_subhead],
        "worst": accs[worst_subhead],
        "best_train_sub_head": matches[best_subhead]
    }


def get_preds_and_targets(config, net_model, val_loader, dataset):
    dataset_size = len(dataset)
    preds = torch.zeros(config.num_sub_heads, dataset_size)
    targets = torch.zeros(dataset_size)
    batch_sz = config.batch_sz

    for b_i, batch in enumerate(val_loader):
        src_x, label = batch
        src_x = src_x.cuda()

        src_x = sobel_process(src_x, config.include_rgb)

        with torch.no_grad():
            src_y = net_model(src_x)  # (batch_size, num_classes)

        start_i = b_i * batch_sz
        end_i = start_i + label.size(0)
        for i in range(config.num_sub_heads):
            preds[i][start_i:end_i] = torch.argmax(src_y[i], dim=1)

        targets[start_i:end_i] = label

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

    config = parse_config()

    train(config)
