import os
import logging
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F






def init_training(rank, args):
    model = get_model(args)  # resnet18(num_classes=10)# ResNet18()
    # model = DDP(model) -- non-private

    #model = DDP(model)

    trainloader, valloader, testloader = get_dataloaders(args)

    if args.steps is not None:
        args.epochs = int(args.steps * args.batch_size / len(trainloader.dataset))

    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay) #momentum=0.9, weight_decay=5e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # lr=0.001)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) #None
    else:
        scheduler = None


    print(
        f"(Initialized model ({type(model).__name__}), "
        f"optimizer ({type(optimizer).__name__}), "
        f"data loader ({type(trainloader).__name__}, len={len(trainloader)})"
    )


    if not args.no_privacy:
        model, optimizer, trainloader, privacy_engine = prepare_DP(args, model, optimizer, trainloader)
        print(
            f"(After privatization: model ({type(model).__name__}), "
            f"optimizer ({type(optimizer).__name__}), "
            f"data loader ({type(trainloader).__name__}, len={len(trainloader)})"
        )

        print(f"(Average batch size per GPU: {int(optimizer.expected_batch_size)}")

        return model, optimizer, trainloader, valloader, testloader, privacy_engine, scheduler
    else:
        return model, optimizer, trainloader, valloader, testloader, None, scheduler


import numpy as np



def test(model, device, testloader, run, args):
    return test_epoch(model, device, testloader, run, args)


def val(model, device, valloader, run, args):
    return val_epoch(model, device, valloader, run,args)

def test_epoch(model, device, testloader, run, args):
    criterion=nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    iterator = tqdm(testloader)
    iterations = 0
    with torch.no_grad():
        for data, targets in iterator:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets)

            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterations = iterations + 1
            iterator.set_postfix({'loss': test_loss / iterations, "acc": 100. * correct / total})

        acc = 100. * correct/float(total)

    print(
        f"\tTest set:"
        f"Acc: {acc:.6f} "
    )


    return acc #accuracy.compute()


def val_epoch(model, device, testloader, run, args):
    criterion=nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    iterator = tqdm(testloader)
    iterations = 0
    with torch.no_grad():
        for data, targets in iterator:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iterations = iterations + 1
            iterator.set_postfix({'loss': test_loss / iterations, "acc": 100. * correct / total})
        acc = 100. * correct/float(total)


    return acc #accuracy.compute()

def train_epoch(model, rank, epoch, optimizer, trainloader, privacy_engine, args, run, elapsed_steps):
    criterion=nn.CrossEntropyLoss()

    #accuracy = Accuracy()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    iterator = tqdm(trainloader)
    iterations = 0

    for data, target in iterator:
        if not args.no_privacy:
            elapsed_steps += 1
            if elapsed_steps >= args.steps + 1:
                return False, None, None


        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        iterations = iterations + 1

        iterator.set_postfix({'loss': train_loss/iterations, "acc": 100* correct/total})

    if not args.no_privacy:
        epsilon = privacy_engine.get_epsilon(args.delta)
        print(
            f"\tTrain Epoch: {epoch} \t"
            f"Loss: {train_loss/iterations:.6f} "
            f"Acc@1: {100* correct/total:.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )

    return True, iterations, 100 * correct/total

def launch(rank, args,checkpoint_base):


    #seed everything
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model, optimizer, trainloader, valloader, testloader, privacy_engine, scheduler = init_training(rank, args)

    run = None
    model.to(rank)
    model.train()

    elapsed_steps = 0
    iterations = 0
    for e in range(args.epochs):
        keep_training, new_its, train_accuracy = train_epoch(model, rank, e, optimizer, trainloader, privacy_engine, args,run,elapsed_steps)
        iterations += new_its

        val_accuracy = val(model, rank, valloader,run, args)

        test_accuracy = test(model, rank, testloader,run, args)

        if not args.no_privacy:
            epsilon = privacy_engine.get_epsilon(delta=1e-5)

            to_save = {
                "epoch": e + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scheduler is not None:
                to_save["scheduler"] = scheduler.state_dict()
            save_checkpoint(to_save, checkpoint_base)
            print("Saving checkpoint in ", checkpoint_base)

            print(
                f"Epoch: {e} \t"
                f"Train Accuracy: {train_accuracy:.2f} | "
                f"Val Accuracy: {val_accuracy:.2f} |"
                f"Test Accuracy: {test_accuracy:.2f} |"
                f"(ε = {epsilon:.2f})"
            )
        else:
            to_save = {
                "epoch": e + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            if scheduler is not None:
                to_save["scheduler"] = scheduler.state_dict()
            save_checkpoint(to_save, checkpoint_base)
            print("Saving checkpoint in ", checkpoint_base)

        if not keep_training:
            break
        if scheduler is None:
            pass
        else:
            scheduler.step()


import torch.multiprocessing as mp

from utils import *

if __name__ == '__main__':

    args = parse_args()
    run_name, checkpoint_base = init_experimentname_folders(args)
    if args.use_gpu:
        rank="cuda:0"
    else:
        rank = "cpu"
    launch(rank,args,checkpoint_base)


