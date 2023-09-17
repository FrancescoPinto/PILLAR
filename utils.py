import argparse


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import Flowers102
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import os
from tqdm import tqdm

from opacus.validators import ModuleValidator


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch DP Training")
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--steps",
        default=4000,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )


    parser.add_argument(
        "-b",
        "--batch_size",
        default=128,
        type=int,
        metavar="N",
        help="logical batch size, physically it won't be this one",
    )
    parser.add_argument(
        "--dim",
        default=None,
        type=int,
        metavar="N",
        help="PCA dimensionality",
    )

    #for binary classification class selection
    parser.add_argument(

        "--cl1",
        default=0,
        type=int,
        metavar="N",
        help="for binary calssification, class 2",
    )
    parser.add_argument(
        "--cl2",
        default=1,
        type=int,
        metavar="N",
        help="for binary calssification, class 2",
    )

    parser.add_argument(
        "-pb",
        "--max_physical_batch_size",
        default=128,
        type=int,
        metavar="N",
        help="physical batch size, actually used in the GPU (check this for memory)",
    )
    parser.add_argument(
        "--momentum", default=0.0, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--accountant", default="prv", type=str, metavar="M", help="SGD momentum"
    )



    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.0,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )

    parser.add_argument(
        "-s", "--seed", default=0, type=int, help="seed for initializing training. "
    )

    parser.add_argument(
        "--fold", default=0, type=int, help="seed for initializing training. "
    )

    parser.add_argument(
        "-c",
        "--max_grad_norm",
        type=float,
        default=1.2,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=50.0,
        help="Target epsilon (default: 50.0)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="Split of the training set to use. Done to ablate sample complexity",
    )
    parser.add_argument(
        "--train_complexity_split",
        type=float,
        default=0.0,
        help="Split of the training set to use. Done to ablate sample complexity",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="r18",
        help="architecture type",
    )


    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="architecture type",
    )


    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="Optimizer type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Dataset name",
    )


    parser.add_argument(
        "--no_privacy",
        action="store_true",
        help="Whether to use differential privacy or not",
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Whether to use differential privacy or not",
    )


    parser.add_argument(
        "--rmp",
        action="store_true",
        help="Whether to use differential privacy or not",
    )

    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Whether to shuffle the trainset",
    )

    parser.add_argument(
        "--bn",
        action="store_true",
        help="Apply batch normalization or not",
    )

    parser.add_argument(
        "--wn",
        action="store_true",
        help="Apply weight normalization or not",
    )

    parser.add_argument(
        "--fine_tune",
        action="store_true",
        help="Apply full body fine-tuning",
    )

    parser.add_argument(
        "--no_aug",
        action="store_true",
        help="Whether to use augmentations",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default = None,
        help="Whether to use differential privacy or not",
    )

    parser.add_argument(
        "--pca_dataset",
        type=str,
        default = None,
        help="Whether to use differential privacy or not",
    )

    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Whether to use differential privacy or not",
    )
    parser.add_argument(
        "--linear_probing",
        action="store_true",
        help="Whether to train just a linear classifier (linear probing)",
    )
    parser.add_argument(
        "--notmodify_forcifar",
        action="store_true",
        help="Whether to train just a linear classifier (linear probing)",
    )
    parser.add_argument(
        "--linear_probing_checkpoint",
        type=str,
        default=None,
        help="Folder with the checkpoint to use for the embeddings",
    )
    parser.add_argument(
        "--force_save_folder",
        type=str,
        default=None,
        help="Folder with the checkpoint to use for the embeddings",
    )
    parser.add_argument(
        "--force_project_name",
        type=str,
        default=None,
        help="Folder with the checkpoint to use for the embeddings",
    )
    parser.add_argument(
        "--clipping",
        type=str,
        default="flat",
        help="Clipping strategy to be used",
    )
    parser.add_argument(
        "--aug_mult",
        type=int,
        default=0,
        help="Clipping strategy to be used",
    )

    parser.add_argument(
        "--aug_diversity",
        type=int,
        default=0,
        help="Clipping strategy to be used",
    )
    parser.add_argument(
        "--not_repeat_interleave",
        action="store_true",
        help="Set to true to use aug_mult for linear probing in embedding files",
    )

    parser.add_argument(
        "--preprocess_like_imagenet",
        action="store_true",
        help="Whether to train just a linear classifier (linear probing)",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Whether to use exponential moving average",
    )
    parser.add_argument(
        "--perc_pca",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Whether to use exponential moving average",
    )


    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="Apply batch normalization or not",
    )

    parser.add_argument(
        "--fine_labels",
        action="store_true",
        help="For c100, whether to use fine labels",
    )


    parser.add_argument(
        "--loss_mode",
        type=str,
        default="CE",
        #can also be BCE for multi-label classification
    )


    parser.add_argument(
        "--compute_percentage_clipping",
        action="store_true",
        help="Apply batch normalization or not",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Pixel size to resize the image before applying cropping",
    )

    parser.add_argument(
        "--crop",
        type=int,
        default=224,
        help="Pixel size to crop to",
    )



    args, unknown = parser.parse_known_args()


    return args

from torch.utils.data import TensorDataset


def get_transforms(dataset, preprocess_like_imagenet, no_aug):
    transform_train = None
    transform_test = None
    if preprocess_like_imagenet:
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)  # from timm library, as used in solo-learn
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        min_scale = 0.08
        max_scale = 1.0
        crop_size = 224
        resize_size = 256

        transform_train = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
    elif dataset=="imagenet_to_cifar": #downscale and upscale imagenet to resemble upscaled cifar
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)  # from timm library, as used in solo-learn
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        min_scale = 0.08
        max_scale = 1.0
        crop_size = 224
        resize_size = 256

        transform_train = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )


    return transform_train, transform_test


from sklearn import random_projection
def get_dataloaders(args,a=None, b=None):
    temp_batch_size = args.batch_size
    if args.linear_probing:

        train_embeds = torch.load(args.linear_probing_checkpoint+f"/{args.dataset}_train_embeds_no_aug")
        train_targets = torch.load(args.linear_probing_checkpoint+f"/{args.dataset}_train_targets_no_aug")
        val_embeds = torch.load(args.linear_probing_checkpoint + f"/{args.dataset}_val_embeds_no_aug")
        val_targets = torch.load(args.linear_probing_checkpoint + f"/{args.dataset}_val_targets_no_aug")
        test_embeds = torch.load(args.linear_probing_checkpoint+f"/{args.dataset}_test_embeds_no_aug")
        test_targets = torch.load(args.linear_probing_checkpoint+f"/{args.dataset}_test_targets_no_aug")


        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        no_aug_embeds = torch.load(args.linear_probing_checkpoint+f"/{args.dataset}_val_embeds_no_aug")

        sc.fit(no_aug_embeds)

        if not args.no_privacy:
            train_targets = torch.load(args.linear_probing_checkpoint + f"/{args.dataset}_train_targets_no_aug")

        original_val_embeds = val_embeds


        train_embeds = sc.transform(train_embeds)
        test_embeds = sc.transform(test_embeds)
        val_embeds = sc.transform(val_embeds)
        original_val_embeds = sc.transform(original_val_embeds)

        from sklearn.decomposition import PCA
        # from diffprivlib.models import PCA
        if args.dim is not None:
            if args.rmp:
                transformer = random_projection.GaussianRandomProjection(args.dim)
                transformer = transformer.fit(train_embeds)
            else:
                transformer = PCA(args.dim)  # 512)# epsilon=2) #GaussianRandomProjection(eps=eps)
                transformer.fit(original_val_embeds[:int(args.perc_pca*original_val_embeds.shape[0])])  # no_aug_embeds)
            train_embeds = transformer.transform(train_embeds)
            test_embeds = transformer.transform(test_embeds)
            val_embeds = transformer.transform(val_embeds)

        train_embeds = torch.tensor(train_embeds).float()
        test_embeds = torch.tensor(test_embeds).float()
        val_embeds = torch.tensor(val_embeds).float()
        train_dataset = TensorDataset(train_embeds, train_targets)
        val_set = TensorDataset(val_embeds, val_targets)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=temp_batch_size, shuffle=not args.no_shuffle)
                            #pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=temp_batch_size)

        test_dataset = TensorDataset(test_embeds, test_targets)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=temp_batch_size)
                            #pin_memory=True)
        return train_loader, val_loader, test_loader
#
    #else:
    transform_train, transform_test = get_transforms(args.dataset, args.preprocess_like_imagenet, args.no_aug)
    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=args.DATA_ROOT, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=args.DATA_ROOT, train=False, download=True, transform=transform_test)
        val_set = None
    elif args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root=args.DATA_ROOT, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=args.DATA_ROOT, train=False, download=True, transform=transform_test)
        val_set = None


    print("val split ", args.val_split, "and val set", val_set)
    if args.val_split > 0 and val_set is None:
        if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "cifar100coarse":
            train_size = 45000
            val_size = 5000
        elif args.dataset == "GermanSigns" or args.dataset == "Dermnet":
            train_size = int(len(trainset) * args.val_split)
            val_size = len(trainset) - train_size #int(len(trainset) * (1 - args.val_split))

        trainset, val_set = torch.utils.data.random_split(trainset, [train_size,val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))


    import numpy as np
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=temp_batch_size, shuffle=not args.no_shuffle, num_workers=args.num_workers)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=100, shuffle=True,  num_workers=args.num_workers)

    return trainloader, valloader, testloader

def save_checkpoint(state, filename_dir, filename="/last_full_epoch.ckpt" ):
    torch.save(state, filename_dir+filename)

import os
import sys
def init_experimentname_folders(args):
    if args.dataset == "cifar10":
        d = "C10"
    elif args.dataset == "cifar100coarse":
        d = "C100coarse"
    else:
        d = args.dataset

    if args.no_privacy:
        run_name = f"{'FT' if args.fine_tune else ''}{args.optimizer}{args.architecture}{d}_lr{args.lr}b{args.batch_size}s{args.steps}wd{args.weight_decay}mom{args.momentum}{'cosineAnn' if args.scheduler else ''}PCA{'full' if args.dim is None else str(args.dim)}" \
                   f"{'no_aug' if args.no_aug else ''}{'ImgNetPreproc' if args.preprocess_like_imagenet else ''}{'_wn' if args.wn else ''}{'_ema' if args.ema else ''}{'_bn' if args.bn else ''}" \
                   f"{'aug_mult' if args.aug_mult > 0 else ''}{args.aug_mult if args.aug_mult > 0 else ''}"
    else:
        run_name = f"{'FT' if args.fine_tune else ''}DP{args.optimizer}{args.architecture}{d}_lr{args.lr}b{args.batch_size}s{args.steps}wd{args.weight_decay}mom{args.momentum}{'cosineAnn' if args.scheduler else ''}PCA{'full' if args.dim is None else str(args.dim)}c{args.max_grad_norm}eps{args.epsilon}del{args.delta}" \
                   f"{'no_aug' if args.no_aug else ''}{'ImgNetPreproc' if args.preprocess_like_imagenet else ''}{'_wn' if args.wn else ''}{'_ema' if args.ema else ''}{'_bn' if args.bn else ''}" \
                   f"{'aug_mult' if args.aug_mult > 0 else ''}{args.aug_mult if args.aug_mult > 0 else ''}"

    if args.val_split > 0:
        run_name += f"val_split{args.val_split}"

    if args.train_complexity_split > 0:
        run_name += f"train_complexity_split{args.train_complexity_split}"

    if args.linear_probing:
        run_name = args.linear_probing_checkpoint + run_name

    b = f"./{args.dataset}"
    DATA_ROOT = "../data/"

    checkpoint_base = b + run_name + f"/seed{args.seed}/"
    args.base = b
    Path(checkpoint_base).mkdir(parents=True, exist_ok=True)
    args.DATA_ROOT = DATA_ROOT
    args.num_classes = args.num_classes #get_numclasses(args)
    args.run_name = run_name
    args.checkpoint_base = checkpoint_base


    return run_name, checkpoint_base

def get_linear_classifier(input_dim, num_classes):
    return torch.nn.Linear(input_dim, num_classes)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x




def get_full_model(args):
    if args.architecture == "resnet50":
        import timm
        model = timm.create_model("resnet50", num_classes=args.num_classes)  # ConvNeXt(10,
    else:
        print(f"your resnet now has {args.num_classes} classes")
        model = resnet18(num_classes=args.num_classes)
        model.return_embeds = True
        if not args.notmodify_forcifar: #Solo-Learn pretrained on ImageNet100 does NOT need this modification
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            model.maxpool = nn.Identity()

    # remove bn and non-privacy preserving layers
    if not args.bn:
        print("==> Fixing model for DP")
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=False)
        print(errors)
        assert len(errors) == 0

    if args.checkpoint_base is not None and args.linear_probing:
        state_dict = torch.load(args.checkpoint_base + "/checkpoint.ckpt")["state_dict"]
        model.load_state_dict(state_dict)
    if args.load_from_checkpoint is not None:
        state_dict = torch.load(args.load_from_checkpoint + "/checkpoint.ckpt")["state_dict"]
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        model.fc = torch.nn.Identity()
        model.load_state_dict(state_dict)
        model.fc = torch.nn.Linear(2048,args.num_classes)

    return model




def get_model(args):
    if args.linear_probing:
        if args.preprocess_like_imagenet == True:
            model = get_linear_classifier(2048 if args.dim is None else args.dim, args.num_classes)
        else:
            model = get_linear_classifier(2048 if args.dim is None else args.dim, args.num_classes)
    else:
        model = get_full_model(args)

    return model


from opacus import PrivacyEngine
from opacus.privacy_engine import forbid_accumulation_hook
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP


def prepare_DP(args, model, optimizer, train_loader):
    if not args.no_privacy:

        privacy_engine = PrivacyEngine(accountant=args.accountant)

        print("size of training set ", len(train_loader.dataset))
        print("batch size ", args.batch_size)




        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(

                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                max_grad_norm=args.max_grad_norm,
                epochs=args.epochs,
        )

        return model, optimizer, train_loader, privacy_engine
    else:
        return model, optimizer, train_loader, None
