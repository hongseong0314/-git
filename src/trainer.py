from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torch_optimizer as optim
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from tqdm import tqdm
from glob import glob
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, grad_scaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn.functional as F

class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.df = args.df
        pass
    
    def setup(self):
        self.args.output_dim = {"cat1":6, "cat2":18, "cat3":128}[self.args.label]
        # model setup
        self.model = self.get_model(model=self.args.model_class, pretrained=self.args.pretrained)
        self.model.to(self.device)
        self.samples_per_cls  = self.df[self.args.label].value_counts().tolist()

        # 옵티마이저 정의
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.args.lr)
        elif self.args.optimizer == 'Lamb':
            self.optimizer = optim.Lamb(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)                                      
        elif self.args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)                                      
        elif self.args.optimizer == 'SAM':
            base_optimizer = torch.optim.SGD  
            self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=self.args.lr, momentum=0.9)

        # # Loss 함수 정의
        # if self.args.weight is not None:
        #     weights = torch.FloatTensor(self.args.weight).to(self.device)
        #     # self.criterion = WeightedFocalLoss(weights=weights)
        #     self.criterion = torch.nn.CrossEntropyLoss(weight=weights).to(self.device)
        # else:
        #     # self.criterion = WeightedFocalLoss(weights=weights)
        #     self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose = True, path='cp.pth')
        
    def fit(self):
        for b in range(self.args.bagging_num):
            print("bagging num : ", b)
             
            previse_name = ''
            best_model_name = ''
            valid_acc_max = 0
            best_loss = np.inf

            if self.args.fold_num <= 1:
                self.train, self.valid = train_test_split(self.df, test_size=0.2, shuffle=True, stratify=self.df[self.args.label])
                train_data_loader, valid_data_loader = self.sampling()
                self.setup()

                iter_per_epoch = len(train_data_loader)
                if self.args.scheduler == "cycle":
                    self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr, steps_per_epoch=iter_per_epoch, 
                                                                    epochs=self.args.epochs)
                elif self.args.scheduler == 'cos':
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.tmax, 
                                                                                    eta_min=self.args.min_lr, verbose=True) 
                self.warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * self.args.warm_epoch)
                
                for epoch in range(self.args.epochs):
                    print("-" * 50)
                    if self.args.scheduler == 'cos':
                        if epoch > self.args.warm_epoch:
                            self.scheduler.step()
                    self.scaler = grad_scaler.GradScaler()
                    label_list, pred_list = self.training(train_data_loader, epoch)
                    
                    # 에폭별 평가 출력
                    train_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                    dis_acc = accuracy_score(np.array(label_list), np.array(pred_list))
                    print("epoch:{}, acc:{}, f1:{}".format(epoch, dis_acc, train_f1))

                    valid_losses, label_list, pred_list = self.validing(valid_data_loader, epoch)
                    valid_acc = accuracy_score(np.array(label_list), np.array(pred_list))
                    valid_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                    print("epoch:{}, acc:{}, f1:{}, loss:{}".format(epoch, valid_acc, valid_f1, np.average(valid_losses)))

                    self.early_stopping(np.average(valid_losses), self.model)

                    # 모델 저장
                    if best_loss > np.average(valid_losses):
                        best_loss = np.average(valid_losses)
                        create_directory(self.args.save_dict)
                        
                        # model_name_bagging_kfold_bestmodel_valid loss로 이름 지정
                        best_model_name = self.args.save_dict + "/model_{}_{}_{:.4f}.pth".format(self.args.CODER, b, best_loss)
                        
                        if isinstance(self.model, torch.nn.DataParallel): 
                            torch.save(self.model.module.state_dict(), best_model_name) 
                        else:
                            torch.save(self.model.state_dict(), best_model_name) 
                        
                        if os.path.isfile(previse_name):
                            os.remove(previse_name)

                        # 갱신
                        previse_name = best_model_name
                    
                    if self.early_stopping.early_stop:
                        print("Early stopping")
                        break
        
            else:
                self.kfold = StratifiedKFold(n_splits=self.args.fold_num, shuffle=True)
                for fold_index, (trn_idx, val_idx) in enumerate(self.kfold.split(self.df, y=self.df[self.args.label]),1):
                    self.train = self.df.iloc[trn_idx,]
                    self.valid  = self.df.iloc[val_idx,]
                    self.setup()
                    train_data_loader, valid_data_loader = self.sampling()

                    iter_per_epoch = len(train_data_loader)
                    if self.args.scheduler == "cycle":
                        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr, steps_per_epoch=iter_per_epoch, 
                                                                        epochs=self.args.epochs)
                    elif self.args.scheduler == 'cos':
                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.tmax, 
                                                                                        eta_min=self.args.min_lr, verbose=True) 
                    self.warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * self.args.warm_epoch)
                    
                    for epoch in range(self.args.epochs):
                        print("-" * 50)
                        if self.args.scheduler == 'cos':
                            if epoch > self.args.warm_epoch:
                                self.scheduler.step()
                        self.scaler = grad_scaler.GradScaler()
                        label_list, pred_list = self.training(train_data_loader, epoch)
                        
                        # 에폭별 평가 출력
                        train_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                        dis_acc = accuracy_score(np.array(label_list), np.array(pred_list))
                        print("epoch:{}, acc:{}, f1:{}".format(epoch, dis_acc, train_f1))

                        valid_losses, label_list, pred_list = self.validing(valid_data_loader, epoch)
                        valid_acc = accuracy_score(np.array(label_list), np.array(pred_list))
                        valid_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                        print("epoch:{}, acc:{}, f1:{}".format(epoch, valid_acc, valid_f1))

                        self.early_stopping(np.average(valid_losses), self.model)

                        # 모델 저장
                        if best_loss > np.average(valid_losses):
                            best_loss = np.average(valid_losses)
                            create_directory(self.args.save_dict)
                            
                            # model_name_bagging_kfold_bestmodel_valid loss로 이름 지정
                            best_model_name = self.args.save_dict + "/model_{}_{}_{}_{:.4f}.pth".format(self.args.CODER, b, fold_index, best_loss)
                            # torch.save(self.model.state_dict(), best_model_name)
                            
                            if isinstance(self.model, torch.nn.DataParallel): 
                                torch.save(self.model.module.state_dict(), best_model_name) 
                            else:
                                torch.save(self.model.state_dict(), best_model_name) 
                            
                            if os.path.isfile(previse_name):
                                os.remove(previse_name)

                            # 갱신
                            previse_name = best_model_name
                        
                        if self.early_stopping.early_stop:
                            print("Early stopping")
                            break
    
    def sampling(self):
        #Dataset 정의
        train_answer, valid_answer = self.train, self.valid
        train_dataset = self.args.Dataset(
                                            train_answer, 
                                            mode='train', 
                                            img_size = self.args.img_size, 
                                            label= self.args.label,
                                            pad=self.args.pad
                                            )
        
        valid_dataset = self.args.Dataset(
                                            valid_answer, 
                                            mode='test', 
                                            img_size = self.args.test_size, 
                                            label= self.args.label,
                                            pad=self.args.pad
                                            )
        
        train_data_loader = DataLoader(
            train_dataset,
            batch_size = self.args.BATCH_SIZE,
            shuffle = True,
        )

        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size = int(self.args.BATCH_SIZE / 2),
            shuffle = False,
        )

        return train_data_loader, valid_data_loader


    def training(self, train_data_loader, epoch):
        self.model.train()
        pred_list, label_list = [], []
        with tqdm(train_data_loader,total=train_data_loader.__len__(), unit="batch") as train_bar:
            for batch_idx, batch_data in enumerate(train_bar):
                train_bar.set_description(f"Train Epoch {epoch}")
                images, labels = batch_data['image'], batch_data['labels']
                images, labels = Variable(images.cuda()), Variable(labels.cuda())

                if epoch <= self.args.warm_epoch:
                    self.warmup_scheduler.step()

                with torch.set_grad_enabled(True):
                    if self.args.amp:
                        self.model.zero_grad(set_to_none=True)
                        
                        with autocast():
                            model_pred  = self.model(images) 
                            loss = CB_loss(labels, 
                                            model_pred, 
                                            self.samples_per_cls, 
                                            self.args.output_dim,
                                            self.args.loss_type, 
                                            self.args.beta, 
                                            self.args.gamma)

                            # loss = self.criterion(model_pred, labels)
                        self.scaler.scale(loss).backward()

                        # Gradient Clipping
                        if self.args.clipping is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clipping)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        self.model.zero_grad(set_to_none=True)
                        model_pred  = self.model(images) 
                        # loss = self.criterion(model_pred, labels)
                        loss = CB_loss(labels, 
                                            model_pred, 
                                            self.samples_per_cls, 
                                            self.args.output_dim,
                                            self.args.loss_type, 
                                            self.args.beta, 
                                            self.args.gamma)
                        loss.backward()
                        
                        if self.args.clipping is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clipping)
                        
                        if self.args.optimizer == 'SAM':
                            # sam optimizer first_steop
                            self.optimizer.first_step(zero_grad=True)
                            # # sam optimizer second_steop
                            CB_loss(labels, 
                                    self.model(images), 
                                    self.samples_per_cls, 
                                    self.args.output_dim,
                                    self.args.loss_type, 
                                    self.args.beta, 
                                    self.args.gamma).backward()
                            # self.criterion(self.model(images), labels).backward()
                            self.optimizer.second_step(zero_grad=True)
                        else:
                            self.optimizer.step()

                    if self.args.scheduler == 'cycle':
                        if epoch > self.args.warm_epoch:
                            self.scheduler.step()

                    # 질병 예측 라벨화
                    model_pred = torch.argmax(model_pred, dim=1).detach().cpu()
                    labels =labels.detach().cpu()
                    
                    pred_list.extend(model_pred.numpy())
                    label_list.extend(labels.numpy())

                batch_acc = (model_pred == labels).to(torch.float).numpy().mean()
                train_bar.set_postfix(train_loss= loss.item(), 
                                        train_batch_acc = batch_acc,
                                        # F1 = train_f1,
                                    )
        return label_list, pred_list
    
    def validing(self, valid_data_loader, epoch):
        valid_dis_acc_list = []
        valid_losses = []
        self.model.eval()
        pred_list, label_list = [], []

        with tqdm(valid_data_loader,total=valid_data_loader.__len__(), unit="batch") as valid_bar:
            for batch_idx, batch_data in enumerate(valid_bar):
                valid_bar.set_description(f"Valid Epoch {epoch}")
                images, labels = batch_data['image'], batch_data['labels']
                images, labels = Variable(images.cuda()), Variable(labels.cuda())

                with torch.no_grad():
                    model_pred  = self.model(images) 
                    
                    # loss 계산
                    # valid_loss = self.criterion(model_pred, labels)
                    valid_loss = CB_loss(labels, 
                                            model_pred, 
                                            self.samples_per_cls, 
                                            self.args.output_dim,
                                            self.args.loss_type, 
                                            self.args.beta, 
                                            self.args.gamma)

                    model_pred = torch.argmax(model_pred, dim=1).detach().cpu()
                    labels =labels.detach().cpu()
                    
                    pred_list.extend(model_pred.numpy())
                    label_list.extend(labels.numpy())

                # accuracy_score(dis_label, dis_out)
                acc = (model_pred == labels).to(torch.float).numpy().mean()

                # print(dis_acc, crop_acc)
                valid_dis_acc_list.append(acc)

                valid_losses.append(valid_loss.item())
                valid_dis_acc = np.mean(valid_dis_acc_list)
        
                valid_bar.set_postfix(valid_loss = valid_loss.item(), 
                                        valid_batch_acc = valid_dis_acc,
                                        )
        return valid_losses, label_list, pred_list

    
    def get_model(self, model, pretrained=False):
        mdl = torch.nn.DataParallel(model(self.args)) if self.args.multi_gpu else model(self.args)
        if not pretrained:
            return mdl
        else:
            print("기학습 웨이트")
            mdl.load_state_dict(torch.load(pretrained))
            return mdl


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 3
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, weights, gamma=2.0):
        super().__init__()
        self.weights = weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.squeeze()
        targets = targets.squeeze()

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.weights[targets]*(1-pt)**self.gamma * BCE_loss

        return F_loss.mean()

def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1).cuda() * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm