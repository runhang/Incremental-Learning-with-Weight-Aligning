import torch
import torchvision
from torchvision.models import vgg16
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
import PIL.Image as Image
import os
import pickle

from dataset import BatchData
from model import Resnet
from cifar100 import Cifar100
from copy import deepcopy


class Trainer:
    def __init__(self, total_cls):
        self.total_cls = total_cls
        self.seen_cls = 0
        self.dataset = Cifar100()
        self.model = Resnet(32,total_cls).cuda()
        print(self.model)
        self.input_transform = Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32,padding=4),
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)
        print("---------------------------------------------")

    def eval(self, valdata):
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(valdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Val Acc: {}".format(acc*100))
        self.model.train()
        print("---------------------------------------------")
        return acc

    # Get learning rate
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, batch_size, epoches, lr, max_size, is_WA):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()

        # Used for Knowledge Distill
        previous_model = None

        dataset = self.dataset
        val_xs = []
        val_ys = []
        train_xs = []
        train_ys = []

        test_accs = []

        for step_b in range(dataset.batch_num):
            print(f"Incremental step : {step_b + 1}")
            
            # Get the train and test data for step b,
            # and split them into train_x, train_y, val_x, val_y
            train, val = dataset.getNextClasses(step_b)
            print(f'number of trainset: {len(train)}, number of testset: {len(test)}')
            train_x, train_y = zip(*train)
            val_x, val_y = zip(*val)
            val_xs.extend(val_x)
            val_ys.extend(val_y)
            train_xs.extend(train_x)
            train_ys.extend(train_y)

            # Transform data and prepare dataloader
            train_data = DataLoader(BatchData(train_xs, train_ys, self.input_transform),
                        batch_size=batch_size, shuffle=True, drop_last=True)
            test_data = DataLoader(BatchData(val_xs, val_ys, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)
            
            # Set optimizer and scheduler
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
            
            # Print the number of classes have been trained
            self.seen_cls += total_cls//dataset.batch_num
            print("seen classes : ", self.seen_cls)
            test_acc = []

            for epoch in range(epoches):
                print("---------------------------------------------")
                print("Epoch", epoch)

                # Print current learning rate
                scheduler.step()
                cur_lr = self.get_lr(optimizer)
                print("Current Learning Rate : ", cur_lr)

                # Train the model with KD
                self.model.train()
                if step_b >= 1:
                    self.stage1_distill(train_data, criterion, optimizer)
                else:
                    self.stage1(train_data, criterion, optimizer)
                
                # Evaluation
                acc = self.eval(test_data)

            if is_WA:
                # Maintaining Fairness
                if step_b >= 1:
                    self.model.weight_align(step_b)

            # deepcopy the previous model used for KD
            self.previous_model = deepcopy(self.model)

            # Evaluate final accuracy at the end of one batch
            acc = self.eval(test_data)
            print(f'Previous accuracies: {acc}')

    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(train_data):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("CE loss :", np.mean(losses))

    def stage1_distill(self, train_data, criterion, optimizer):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        beta = (self.seen_cls - 20)/ self.seen_cls
        print("classification proportion 1-beta = ", 1-beta)
        for i, (image, label) in enumerate(train_data):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            with torch.no_grad():
                pre_p = self.previous_model(image)
                pre_p = F.softmax(pre_p[:,:self.seen_cls-20]/T, dim=1)
            logp = F.log_softmax(p[:,:self.seen_cls-20]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], label)
            loss = loss_soft_target * T * T + (1-beta) * loss_hard_target
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("KD loss :", np.mean(distill_losses), "; CE loss :", np.mean(ce_losses))

    
