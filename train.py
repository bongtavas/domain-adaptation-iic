from models.net5g_two_head import ClusterNet5gTwoHead
from torch.utils.data import DataLoader
from utils.IID_losses import IID_loss

import numpy as np
import argparse
import time
import logging
import torch
import os
from torchvision import datasets
from torchvision import transforms
from utils.transforms import sobel_process, custom_greyscale_to_tensor
from utils.data import create_dataloaders, get_preds_and_targets, get_preds_actual_class
from utils.train_utils import get_opt
from utils.eval_metrics import hungarian_match, accuracy

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

    # Dataset Settings
    parser.add_argument("--dset_A_name", default="MNIST")
    parser.add_argument("--dset_B_name", default="SVHN")
    parser.add_argument("--dset_B_all", default=False, action="store_true")
    parser.add_argument("--num_dataloaders", type=int, default=5)

    # Model Settings
    parser.add_argument("--head_A_first", default=True, action="store_true")
    parser.add_argument("--head_A_epochs", type=int, default=1)
    parser.add_argument("--head_B_epochs", type=int, default=2)

    parser.add_argument("--output_k_A", type=int, default=50)
    parser.add_argument("--output_k_B", type=int, default=10)

    # Transforms
    parser.add_argument("--input_sz", type=int, default=32)
    parser.add_argument("--rand_crop_sz", type=int, default=20)
    parser.add_argument("--rot_val", type=float, default=25)
    parser.add_argument("--include_rgb", default=False, action="store_true")

    config = parser.parse_args()

    config.output_k = config.output_k_B

    if config.include_rgb:
        config.in_channels = 5
    else:
        config.in_channels = 2

    return config


def train(config):

    dataloaders_head_A, dataloaders_head_B, src_test_loader, dest_test_loader = create_dataloaders(config)

    model = ClusterNet5gTwoHead(config)

    model.cuda()

    model = torch.nn.DataParallel(model)

    optimiser = get_opt(config.opt)(model.module.parameters(), lr=config.lr)

    best_acc = 0.

    model_save_path = os.path.join(config.results_path, "iic-adapt.pt")

    heads = ["A", "B"]

    head_epochs = {}
    head_epochs["A"] = config.head_A_epochs
    head_epochs["B"] = config.head_B_epochs
    
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):

        for head_i in range(2):
            head = heads[head_i]

            if head == "A":
                dataloaders = dataloaders_head_A
            elif head == "B":
                dataloaders = dataloaders_head_B

            for head_i_epoch in range(head_epochs[head]):
                batch_i = 0
                for dataloader in dataloaders:
                    for batch in dataloader:
                        train_model(model, head, batch, batch_i, epoch, optimiser, start_time)
                        batch_i += 1
        dest_stats = evaluate(model, config, src_test_loader, dest_test_loader)
        if dest_stats["best"] > best_acc:
            best_acc = dest_stats["best"]
            logging.info("Accuracy improved, saving model")
            torch.save(model.state_dict(), model_save_path)

        elapsed = time.time() - start_time
        logging.info("Elapsed Time {%4f} " % (elapsed,))


def train_model(model, head, batch, batch_i, epoch, optimiser, start_time):
    model.module.zero_grad()
    src_x, dest_x, _ = batch
    src_x = src_x.cuda()
    dest_x = dest_x.cuda()

    src_x = sobel_process(src_x, config.include_rgb)
    dest_x = sobel_process(dest_x, config.include_rgb)

    z = model(src_x, head=head)
    z_prime = model(dest_x, head=head)

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

    if batch_i % 40 == 0:
        logging.info("=== Epoch {%s} Head {%s} Batch: {%d} Avg Loss: {%f}" %
                     (str(epoch), head, batch_i, avg_loss_batch.item()))

    avg_loss_batch.backward()
    optimiser.step()


def print_stats(stats):
    for key in stats:
        print("{} : {}".format(key, stats[key]))
    print()


def evaluate(net_model, config, src_test_loader, dest_test_loader):
    net_model.eval()
    src_stats = eval_model(net_model, src_test_loader)
    dest_stats = eval_model(net_model, dest_test_loader)
    net_model.train()

    print("==== Source Stats ====")
    print_stats(src_stats)
    print("==== Dest Stats ====")
    print_stats(dest_stats)

    return dest_stats


def eval_model(net_model, dataloader):
    preds, targets = get_preds_and_targets(config, net_model, dataloader)
    preds = preds.cuda()
    targets = targets.cuda()

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
        "best": accs[best_subhead],
        "worst": accs[worst_subhead],
        "avg": np.mean(accs),
        "std": np.std(accs),
        "best_train_sub_head": matches[best_subhead],
        "accs": accs,
    }


if __name__ == "__main__":

    config = parse_config()
    train(config)
