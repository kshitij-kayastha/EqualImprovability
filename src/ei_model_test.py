import tqdm
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from types import SimpleNamespace
from typing import Callable
from torch.utils.data import DataLoader, random_split
from copy import deepcopy
from abc import ABC, abstractmethod

from data import FairnessDataset
from ei_effort import Effort
from ei_utils import model_performance


def fair_batch_proxy(Z: torch.tensor, Y_hat_max: torch.tensor, loss_fn: Callable):
    proxy_val = torch.tensor(0.)
    loss_mean = loss_fn(Y_hat_max, torch.ones(len(Y_hat_max)))
    loss_z = torch.zeros(len(torch.unique(Z)))
    for z in torch.unique(Z):
        z = int(z)
        group_idx = (Z == z)
        if group_idx.sum() == 0:
            continue
        loss_z[z] = loss_fn(Y_hat_max[group_idx], torch.ones(group_idx.sum()))
        proxy_val += torch.abs(loss_z[z]-loss_mean)
    return proxy_val

def covariance_proxy(Z: torch.tensor, Y_hat_max: torch.tensor, loss_fn: Callable = None):
    proxy_val = torch.square(torch.mean((Z-Z.mean())*Y_hat_max))
    return proxy_val

class EIModel():
    def __init__(self, model: nn.Module, proxy: Callable, effort: Effort, tau: float = 0.5) -> None:
        self.model = model
        self.proxy = proxy
        self.effort = effort
        self.tau = tau
        self.train_history = SimpleNamespace()
        
    def train(self, 
              dataset: FairnessDataset, 
              lamb: float,
              alpha: float = 0.,
              lr: float = 1e-2,
              n_epochs: int = 100,
              batch_size: int = 1024,
              abstol: float = 1e-7
              ):

        generator = torch.Generator().manual_seed(0)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        
        pred_losses = []
        fair_losses = []
        accuracies = []
        ei_disparities = []
        
        loss_fn = torch.nn.BCELoss(reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        for epoch in tqdm.trange(n_epochs, desc=f"{'Robust ' if alpha > 0 else ''}Training [alpha={alpha:.2f}; lambda={lamb:.2f}; delta={self.effort.delta:.2f}]", unit="epochs", colour='#0091ff'):
            
            batch_pred_losses = []
            batch_fair_losses = []
            
            for _, (X_batch, Y_batch, Z_batch) in enumerate(data_loader):
                Y_hat = self.model(X_batch).reshape(-1)
                
                batch_loss = 0.
                
                batch_pred_loss = loss_fn(Y_hat, Y_batch)
                batch_loss += (1-lamb)*batch_pred_loss
                
                batch_fair_loss = 0
                if torch.sum(Y_hat<self.tau) > 0:
                    X_batch_e = X_batch[(Y_hat<self.tau), :]
                    Z_batch_e = Z_batch[(Y_hat<self.tau)]
                    
                    X_hat_max = self.effort(self.model, dataset, X_batch_e)
                    
                    if alpha == 0:
                        Y_hat_max = self.model(X_hat_max).reshape(-1)
                    # PGA
                    else:
                        model_adv = deepcopy(self.model)
                        optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-3, maximize=True)
                        pga_loss_fn = torch.nn.BCELoss(reduction='mean')
                        
                        for module in model_adv.layers:
                            if hasattr(module, 'weight'):
                                weight_min = module.weight.data - alpha
                                weight_max = module.weight.data + alpha
                            if hasattr(module, 'bias'):
                                bias_min = module.bias.data - alpha
                                bias_max = module.bias.data + alpha
                                
                        loss_diff = 1.
                        pga_fair_loss = torch.tensor(0.)
                        
                        while loss_diff > abstol:
                        # for _ in range(20):
                            prev_loss = pga_fair_loss.clone().detach()
                            
                            Y_hat_max_pga = model_adv(X_hat_max).reshape(-1)
            
                            pga_fair_loss = self.proxy(Z_batch_e, Y_hat_max_pga, pga_loss_fn)
            
                            optimizer_adv.zero_grad()
                            pga_fair_loss.backward()
                            optimizer_adv.step()
            
                            loss_diff = (prev_loss - pga_fair_loss).abs()
                            
                            for module in model_adv.layers:
                                if hasattr(module, 'weight'):
                                    module.weight.data = module.weight.data.clamp(weight_min, weight_max)
                                if hasattr(module, 'bias'):
                                    module.bias.data = module.bias.data.clamp(bias_min, bias_max)
                        
                        Y_hat_max = model_adv(X_hat_max).reshape(-1)
                    
                    batch_fair_loss = self.proxy(Z_batch_e, Y_hat_max, loss_fn)
                
                batch_loss += lamb*batch_fair_loss
                
                optimizer.zero_grad()
                if torch.isnan(batch_loss).any():
                    continue
                batch_loss.backward()
                optimizer.step()
        
                batch_pred_losses.append(batch_pred_loss.item())
                if hasattr(batch_fair_loss,'item'):
                    batch_fair_losses.append(batch_fair_loss.item())
                else:
                    batch_fair_losses.append(batch_fair_loss)
            
            pred_losses.append(np.mean(batch_pred_losses))
            fair_losses.append(np.mean(batch_fair_losses))

            Y_hat_train = self.model(dataset.X).reshape(-1).detach().numpy()
            X_hat_max_train = self.effort(self.model, dataset, dataset.X)
            Y_hat_max_train = self.model(X_hat_max_train).reshape(-1).detach().numpy()
            
            accuracy, ei_disparity = model_performance(dataset.Y.detach().numpy(), dataset.Z.detach().numpy(), Y_hat_train, Y_hat_max_train, self.tau)
            
            accuracies.append(accuracy)
            ei_disparities.append(ei_disparity)

        self.train_history.accuracy = accuracies
        self.train_history.p_loss = pred_losses
        self.train_history.f_loss = fair_losses
        self.train_history.ei_disparity = ei_disparities
        return self
        
        
    def predict(self,
                dataset: FairnessDataset,
                alpha: float,
                abstol: float = 1e-7
                ):
        
        Y_hat = self.model(dataset.X).reshape(-1)
        X_e = dataset.X[(Y_hat<self.tau).reshape(-1),:]
        Z_e = dataset.Z[(Y_hat<self.tau)]
        
        X_hat_max = self.effort(self.model, dataset, X_e)
        
        model_adv = deepcopy(self.model)
        optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-3, maximize=True)
        pga_loss_fn = torch.nn.BCELoss(reduction='mean')
        
        for module in model_adv.layers:
            if hasattr(module, 'weight'):
                weight_min = module.weight.data - alpha
                weight_max = module.weight.data + alpha
            if hasattr(module, 'bias'):
                bias_min = module.bias.data.item() - alpha
                bias_max = module.bias.data.item() + alpha
        
        loss_diff = 1.
        fair_loss = torch.tensor(0.)
        while loss_diff > abstol:
            prev_loss = fair_loss.clone().detach()
            Y_hat_max = model_adv(X_hat_max).reshape(-1)
            
            fair_loss = self.proxy(Z_e, Y_hat_max, pga_loss_fn)
            
            optimizer_adv.zero_grad()
            fair_loss.backward()
            optimizer_adv.step()
            
            loss_diff = (prev_loss - fair_loss).abs()
            
            for module in model_adv.layers:
                if hasattr(module, 'weight'):
                    module.weight.data = module.weight.data.clamp(weight_min, weight_max)
                if hasattr(module, 'bias'):
                    module.bias.data = module.bias.data.clamp(bias_min, bias_max)
                    
        self.model_adv = model_adv
        Y_hat = Y_hat.detach().float().numpy()
        X_hat_max = self.effort(self.model, dataset, dataset.X)
        Y_hat_max = self.model_adv(X_hat_max).reshape(-1).detach().float().numpy()
        
        return Y_hat, Y_hat_max, fair_loss