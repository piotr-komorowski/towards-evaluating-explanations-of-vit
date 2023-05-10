#!/usr/bin/env python3
import argparse
import logging
import logging.handlers
import os
# import sys
import pdb
import random
import re
import shutil
import multiprocessing
from PIL import Image

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader

from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP_base
from baselines.ViT.ViT_LRP import vit_large_patch16_224 as vit_LRP_large


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.shape[0]
        pred = output.argmax(dim=1)
        correct = pred.eq(target)
        acc = correct.float().sum().mul(100.0 / batch_size)
        return acc


def preprocess(args):
    print('prepare model')
    if args.model == 'base':
        model = vit_LRP_base(pretrained=args.pretrained)
    elif args.model == 'large':
        model = vit_LRP_large(pretrained=args.pretrained)
    else:
        raise ValueError(f'Unknown model {args.model}')
    model.head = nn.Linear(model.head.in_features, 3) # change number of outputs to match number of classes
    return model.to(args.device)


def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    filename = f'checkpoint_epoch_{state["epoch"]}.pth.tar'
    path = os.path.join(directory, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(directory, 'model_best.pth.tar'))


def train(args, model):
    print('Calculating mean and std')
    preprocess_dataset = datasets.ImageFolder(
        args.train_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))

    preprocess_loader = DataLoader(preprocess_dataset, batch_size=args.train_batch_size*16,
                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                             drop_last=False)
    preprocess_data = next(iter(preprocess_loader))
    mean = torch.mean(preprocess_data[0], dim=(0, 2, 3))
    std = torch.std(preprocess_data[0], dim=(0, 2, 3))
    print(f'Mean: {mean}, std: {std}')

    print('Training data loading')
    normalize = transforms.Normalize(mean=mean, std=std)
    train_dataset = datasets.ImageFolder(
    args.train_dir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ]))
    print(f'Classes: {train_dataset.class_to_idx}')
    print(f'arguments here: {vars(args)}')
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                             num_workers=args.num_workers, pin_memory=True, shuffle=True,
                             drop_last=False)

    print('Training data loading completed')

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Clear file
    with open(args.temp_output, 'r+') as f:
        f.truncate(0)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    else:
        raise ValueError(f"The directory {args.results_dir} already exists.")

    
    best_val_acc = torch.tensor(0) # as tensor because validation acc will be a tensor
    

    for epoch in range(args.num_epochs):
        model.train()
        print(f'epoch {epoch}/{args.num_epochs-1}')
        train_acc = 0
        train_count = 0
        
        for i, (image, target) in enumerate(train_loader):
            
            image, target = image.to(args.device), target.to(args.device)

            optimizer.zero_grad()
            output = model(image)

            loss = loss_func(output.squeeze(), target)
            acc = accuracy(output, target)
            train_count += image.shape[0]
            train_acc += acc * image.shape[0]
            train_acc_avg = train_acc / train_count
            
            loss.backward()
            optimizer.step()

            if i % 100 == 0 and i != 0:
                with open(args.temp_output, 'a') as f:
                    f.write((f"Train, step #{i}/{len(train_loader)}, "
                            f"accuracy {train_acc_avg:.3f}, "
                            f"loss {loss:.3f}, \n"))
                print((f"Train, step #{i}/{len(train_loader)}, "
                            f"accuracy {train_acc_avg:.3f}, "
                            f"loss {loss:.3f}, \n"))

        val_acc = validate(args, model)
        
        is_best = val_acc > best_val_acc
        log_val = f'Val acc improved from {round(best_val_acc.item(), 3)} to {round(val_acc.item(), 3)}' if is_best \
            else f'Val acc has not improved form {round(best_val_acc.item(), 3)}'
        best_val_acc = max(val_acc, best_val_acc)
        print(log_val)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_val_acc': best_val_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, 
        args.results_dir)


    log = '\n'.join([
        f'training results',
        f'- Accuracy: {train_acc_avg:.4f}'])

    print('train finish')
    print(log)


def validate(args, model):
    normalize = transforms.Normalize(mean=[0.58, 0.58, 0.58], std=[0.21, 0.21, 0.21])

    val_dataset = datasets.ImageFolder(
        args.val_dir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size,
                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                             drop_last=False)

    model.eval()

    loss_func = nn.CrossEntropyLoss()

    val_acc = 0
    val_count = 0

    with torch.no_grad():
        for i, (image, target) in enumerate(val_loader):
            image, target = image.to(args.device), target.to(args.device)
            output = model(image)
            
            loss = loss_func(output.squeeze(), target)
            acc = accuracy(output, target)
            val_count += image.shape[0]
            val_acc += acc * image.shape[0]
            val_acc_avg = val_acc / val_count


            if i % 100 == 0 and i != 0:
                with open(args.temp_output, 'a') as f:
                    f.write((f"Val, step #{i}/{len(val_loader)}, "
                            f"accuracy {val_acc_avg:.3f}, "
                            f"loss {loss:.3f}, \n"))
                print((f"Val, step #{i}/{len(val_loader)}, "
                            f"accuracy {val_acc_avg:.3f}, "
                            f"loss {loss:.3f}, \n"))




    log = '\n'.join([
        f'val results',
        f'- Accuracy: {val_acc_avg:.4f}'])

    print('val finish')
    print(log)

    return val_acc_avg

def test(args, model):
    normalize = transforms.Normalize(mean=[0.58, 0.58, 0.58], std=[0.21, 0.21, 0.21])
    test_dataset = datasets.ImageFolder(
        args.test_dir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                             num_workers=args.num_workers, pin_memory=True, shuffle=False,
                             drop_last=False)

    
    loaded_model = torch.load(os.path.join(args.results_dir, 'model_best.pth.tar'))
    model.load_state_dict(loaded_model['state_dict'])
    print(f'loading model from epoch {loaded_model["epoch"]} with val acc {loaded_model["best_val_acc"]:.3f}')
    model.eval()
    loss_func = nn.CrossEntropyLoss()

    print('testing')

    test_acc = 0
    test_count = 0
    with torch.no_grad():
        for i, (image, target) in enumerate(test_loader):
            image, target = image.to(args.device), target.to(args.device)
            output = model(image)
            loss = loss_func(output.squeeze(), target)
            acc = accuracy(output, target)
            test_count += image.shape[0]
            test_acc += acc * image.shape[0]
            test_acc_avg = test_acc / test_count
            

            if i % 100 == 0:
                with open(args.temp_output, 'a') as f:
                    f.write((f"test, step #{i}/{len(test_loader)}, "
                            f"accuracy {test_acc_avg:.3f}, "
                            f"loss {loss:.3f}, \n"))
                print((f"Test, step #{i}/{len(test_loader)}, "
                            f"accuracy {test_acc_avg:.3f}, "
                            f"loss {loss:.3f}, "))

    log = '\n'.join([
        f'# Test Result',
        f'- acc: {test_acc_avg:.4f}'])
    print('test finish')
    print(log)




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='base',
                        choices=['base', 'large'],
                        help='model type')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='directory path of train dataset')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='directory path of val dataset')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='directory path of test dataset')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='batch size of training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='batch size of validation')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='batch size of inference')
    parser.add_argument('--results_dir', type=str, default='results_sched_no_pretrain_aug',
                        help='directory path of result')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='maximum number of epochs for training')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pretrained', type=int, default=0,
                        choices=[0, 1],
                        help='whether the model should be pretrained on imagenet')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--temp_output', type=str, default='output.txt',
                        help='file for temporal output')
                        

    args = parser.parse_args()


    model = preprocess(args)
    train(args, model)
    test(args, model)
    
