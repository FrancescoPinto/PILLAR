# PILLAR
Official Implementation of PILLAR: How to make semi-private learning more effective

## Getting Checkpoint and Setting Up The Environment
Download the ResNet50 trained with SL from:

`https://drive.google.com/file/d/1EvzpxvgbEaUU1OxxUJvjdu_Yp1cxwaHD/view?usp=sharing`

Place it in a subfolder called `./resnet50_sl/`

(FOR GPU INSTALL) Create your pip environment and run `pip install -r requirements.txt`. 

(FOR CPU INSTALL) First install CPU pytorch and remove corresponding line in the requirements file. 

We assume `cuda:0` is always used. If cpus are required, please modify that string in `trian.py` or `prepare_rescaled_cifar.py` to `cpu`. 

## Extracting the Embeddings into Files
To make training more efficient, we avoid performing a forward pass through the whole network.
We make a single pass to save embeddings into files.

To get the CIFAR10 and CIFAR100 embeddings, run the command:

`python prepare_rescaled_cifar.py`

This will create train, val and test files in the `./resnet50_sl/` subfolder.

## Running our method with optimal hyperparameters
Run whichever line of the `commands.sh` file you may want to execute. To use different seeds 
modify the parameter `-s` with an integer.
Two batches of commands are available in the commands.sh file, one for CIFAR-10, the other for CIFAR-100.

The provided code is sufficient to reproduce Figure 4a (both DP-SGD and PILLAR).

## Cite our work
If you found this repository useful, please cite our code as:


```
@article{pinto2023pillar,
  title={PILLAR: How to make semi-private learning more effective},
  author={Pinto, Francesco and Hu, Yaxi and Yang, Fanny and Sanyal, Amartya},
  journal={arXiv preprint arXiv:2306.03962},
  year={2023}
}
```
