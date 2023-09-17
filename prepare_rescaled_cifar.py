from resnet50_backbone import ResidualNet
def create_embedding_files(args, model, device):
    # iterate over the training and test set to save the embeddings
    args.linear_probing = False
    temp_split = args.train_complexity_split
    args.train_complexity_split = 0
    train_loader, val_loader, test_loader = get_dataloaders(args)
    args.linear_probing = True
    args.train_complexity_split = temp_split
    #must load a checkpoint
    model.eval()
    model = model.to(device)
    train_embeds = []
    test_embeds = []
    val_embeds = []
    train_targets = []
    test_targets = []
    val_targets = []
    with torch.no_grad():
        print("Creating embeddings for the training set")
        print("train_loader size ", len(train_loader))
        for i, (images, target) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            embeds = model(images)
            train_embeds.append(embeds.squeeze().cpu())
            train_targets.append(target)
        print("Creating embeddings for the val set")
        print("val_loader size ", len(val_loader))
        for i, (images, target) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            embeds = model(images)
            val_embeds.append(embeds.squeeze().cpu())
            val_targets.append(target)
        print("Creating embeddings for the test set")
        print("test_loader size ", len(test_loader))
        for i, (images, target) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            embeds = model(images)
            test_embeds.append(embeds.squeeze().cpu())
            test_targets.append(target)

    train_embeds = torch.cat(train_embeds, dim=0)
    test_embeds = torch.cat(test_embeds, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    val_embeds = torch.cat(val_embeds, dim=0)
    val_targets = torch.cat(val_targets, dim=0)

    print("val_embeds size", val_embeds.shape)
    print("train_embeds size", train_embeds.shape)

    print("Saving the results in: ",args.checkpoint_base)
    torch.save(train_embeds, args.checkpoint_base+f"/{args.dataset}_train_embeds_no_aug")#_highres")
    torch.save(test_embeds, args.checkpoint_base+f"/{args.dataset}_test_embeds_no_aug")#_highres")
    torch.save(val_embeds, args.checkpoint_base+f"/{args.dataset}_val_embeds_no_aug")#_highres")
    torch.save(train_targets, args.checkpoint_base+f"/{args.dataset}_train_targets_no_aug")#_highres")
    torch.save(test_targets, args.checkpoint_base+f"/{args.dataset}_test_targets_no_aug")#_highres")
    torch.save(val_targets, args.checkpoint_base+f"/{args.dataset}_val_targets_no_aug")#_highres")

import os
import torch.distributed as dist
import logging
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP
from opacus import PrivacyEngine

from tqdm import tqdm
from torchmetrics import Accuracy, CalibrationError
import random
import numpy as np



def init(args):
    #timm checkpoints
    if args.checkpoint_base.find("SSL") < 0:
        if args.checkpoint_base.find("32x32") < 0:
            import timm
            model = timm.create_model("resnet50", num_classes=args.num_classes)
            state_dict = torch.load(args.checkpoint_base + "/checkpoint.ckpt")["state_dict"]
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
            model.fc = torch.nn.Identity()
            model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(args.checkpoint_base + "/checkpoint.ckpt")["state_dict"]

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.find("module.") < 0:
                    break
                else:
                    name = k.replace("module.", "")
                    new_state_dict[name] = v
            state_dict = new_state_dict
            model = ResidualNet(50)
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
            model.fc = torch.nn.Identity()
            model.load_state_dict(state_dict)


    else:
        #solo-learn checkpoints
        from torchvision.models import resnet50
        model = resnet50()
        new_state_dict = {}
        state_dict = torch.load(args.checkpoint_base + "/checkpoint.ckpt")["state_dict"]
        for k, v in state_dict.items():  # only care about the backbone
            if not k.startswith("backbone."):
                continue
            new_state_dict[k.replace("backbone.", "")] = v
        state_dict = new_state_dict
        model.fc = torch.nn.Identity()
        model.load_state_dict(state_dict)

    args.linear_probing = False
    trainloader, valloaders, testloader = get_dataloaders(args)
    args.linear_probing = True

    return model, trainloader, valloaders, testloader


import numpy as np


def launch(args, device):

    #seed everything
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.no_aug = True #no augmentations in preliminary version

    model, _, _, _ = init(args)

    model.to(device)
    create_embedding_files(args, model, device)



import torch.multiprocessing as mp

world_size = torch.cuda.device_count()
from utils import *

if __name__ == '__main__':

    args = parse_args()
    if args.use_gpu:
        device="cuda:0"
    else:
        device="cpu"

    checkpoint_bases = [
    #resnet50
    "./resnet50_sl/",
    ]


    DATA_ROOT = "../data/"
    args.architecture = "resnet50"
    for checkpoint_base in checkpoint_bases:
        Path(checkpoint_base).mkdir(parents=True, exist_ok=True)
        args.DATA_ROOT = DATA_ROOT
        args.checkpoint_base = checkpoint_base
        args.val_split = 0.9

        args.dataset = "cifar10"
        args.preprocess_like_imagenet = True
        launch(args, device)

        args.dataset = "cifar100"
        args.preprocess_like_imagenet = True
        launch(args, device)


