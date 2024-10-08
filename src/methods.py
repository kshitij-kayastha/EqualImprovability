import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from typing import Callable
from torch.autograd import grad
from types import SimpleNamespace
from scipy.optimize import linprog
from torch.utils.data import DataLoader

from src.model import LR, NN
from src.effort import Effort
from src.data import FairnessDataset
from src.utils import model_performance


a = 0.4920
b = 0.2887
c = 1.1893
Q_function = lambda x: torch.exp(-a*x**2 - b*x - c) 

def Huber(x, delta):
    if x.abs()<delta:
        return (x**2)/2
    return delta*(x.abs()-delta/2)

def grad_Huber(x, delta):
    if x.abs()>delta:
        if x>0:
            return delta
        else:
            return -delta
    return x

def CDF_tau(Y_hat, h=0.01, tau=0.5):
    m = len(Y_hat)
    Y_tilde = (tau-Y_hat)/h
    sum_ = torch.sum(Q_function(Y_tilde[Y_tilde>0])) \
           + torch.sum(1-Q_function(torch.abs(Y_tilde[Y_tilde<0]))) \
           + 0.5*(len(Y_tilde[Y_tilde==0]))
    return sum_/m

def kde_proxy(Z: torch.Tensor, Y_hat_max: torch.Tensor, tau: float = 0.5):
    h = 0.01
    delta_huber = 0.5
    pi = torch.tensor(np.pi)
    phi = lambda x: torch.exp(-0.5*x**2)/torch.sqrt(2*pi)
    
    fair_loss = torch.tensor(0.)
    Pr_Y_tilde_1 = CDF_tau(Y_hat_max.detach(),h,tau)
    for z in [0,1]:
        if torch.sum(Z==z)==0:
            continue
        Pr_Y_tilde_1_Z = CDF_tau(Y_hat_max.detach()[Z==z], h, tau)
        m_z = Z[Z==z].shape[0]
        m = Z.shape[0]

        Delta_z = Pr_Y_tilde_1_Z-Pr_Y_tilde_1
        Delta_z_grad = torch.dot(phi((tau-Y_hat_max.detach()[Z==z])/h).view(-1), 
                                    Y_hat_max[Z==z].view(-1))/h/m_z
        Delta_z_grad -= torch.dot(phi((tau-Y_hat_max.detach())/h).view(-1), 
                                    Y_hat_max.view(-1))/h/m

        Delta_z_grad *= grad_Huber(Delta_z, delta_huber)
        fair_loss += Delta_z_grad
    
    return fair_loss

def covariance_proxy(Z: torch.Tensor, Y_hat_max: torch.Tensor, tau: float = 0.5):
    fair_loss = torch.square(torch.mean((Z-Z.mean())*Y_hat_max))
    return fair_loss

def fair_batch_proxy_bce(Z: torch.Tensor, Y_hat_max: torch.Tensor, tau: float = 0.5):
    fair_loss = torch.tensor(0.)
    loss_fn = torch.nn.BCELoss(reduction='mean')

    loss_mean = loss_fn(Y_hat_max, torch.ones(len(Y_hat_max)))
    for z in [0,1]:
        z = int(z)
        group_idx = (Z==z)
        if group_idx.sum() == 0:
            loss_z = torch.tensor(0.).float()
        else:
            loss_z = loss_fn(Y_hat_max[group_idx], torch.ones(group_idx.sum()))
        fair_loss += torch.abs(loss_z - loss_mean)
    return fair_loss

def fair_batch_proxy_mse(Z: torch.Tensor, Y_hat_max: torch.Tensor, tau: float = 0.5):
    fair_loss = torch.tensor(0.)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    loss_mean = loss_fn(Y_hat_max, torch.ones(len(Y_hat_max)))
    for z in [0,1]:
        z = int(z)
        group_idx = (Z==z)
        if group_idx.sum() == 0:
            loss_z = torch.tensor(0.).float()
        else:
            loss_z = loss_fn(Y_hat_max[group_idx], torch.ones(group_idx.sum()))
        fair_loss += torch.abs(loss_z - loss_mean)
    return fair_loss
    
    
class EIModel:
    def __init__(self, model: LR | NN, proxy: Callable, effort: Effort, tau: float = 0.5) -> None:
        self.model = model
        self.proxy = proxy
        self.effort = effort
        self.tau = tau
        self.train_history = SimpleNamespace()
        
    def get_model_adv(self, X_hat_max: torch.Tensor, Z: torch.Tensor, alpha: float, n_epochs: int):
        for module in self.model.layers:
            if hasattr(module, 'weight'):
                weight_min = module.weight.data - alpha
                weight_max = module.weight.data + alpha
            if hasattr(module, 'bias'):
                bias_min = module.bias.data - alpha
                bias_max = module.bias.data + alpha
        
        model_adv = deepcopy(self.model).xavier_init().clamp((weight_min, weight_max), (bias_min, bias_max))
        # model_adv = deepcopy(self.model).randn_noise().clamp((weight_min, weight_max), (bias_min, bias_max))
        optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-3, maximize=True)
                
        fair_loss = torch.tensor(0.)
        for _ in range(int(n_epochs)):
            optimizer_adv.zero_grad()
            
            Y_hat_max = model_adv(X_hat_max).reshape(-1)
            fair_loss = self.proxy(Z, Y_hat_max)

            fair_loss.backward()
            optimizer_adv.step()
            
            model_adv.clamp((weight_min, weight_max), (bias_min, bias_max))
                    
        return model_adv
        
    def train(self, 
              dataset: FairnessDataset, 
              lamb: float,
              alpha: float = 0.,
              lr: float = 1e-3,
              n_epochs: int = 100,
              batch_size: int = 1024,
              abstol: float = 1e-5,
              pga_n_epochs: int = 50
              ):
        
        lamb = torch.tensor(lamb).float()
        
        # initialize arrays to track training metrics
        self.train_history.pred_loss = []
        self.train_history.fair_loss = []
        self.train_history.total_loss = []
        self.train_history.theta_adv = []
        self.train_history.theta = []
        
        # batch loader
        generator = torch.Generator().manual_seed(0)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        
        loss_fn = torch.nn.BCELoss(reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        loss_diff = 1.
        prev_loss = torch.tensor(0.)
        
        for epoch in tqdm.trange(n_epochs, desc=f"Training [alpha={alpha:.3f}; lambda={lamb:.5f}; delta={self.effort.delta:.3f}]", unit="epochs", colour='#0091ff'):
            
            # initialize arrays to track batch metrics
            batch_losses = []
            batch_pred_losses = []
            batch_fair_losses = []

            curr_alpha = alpha * (epoch / n_epochs)
            for _, (X_batch, Y_batch, Z_batch) in enumerate(data_loader):
                # get predictions with theta
                Y_hat = self.model(X_batch).reshape(-1)

                # calculate prediction loss
                batch_pred_loss = loss_fn(Y_hat, Y_batch)
                batch_loss = (1-lamb)*batch_pred_loss
                
                batch_fair_loss = torch.tensor(0.)
                if torch.sum(Y_hat<self.tau) > 0:
                    optimizer.zero_grad()
                    
                    # instances that received a negative label from theta
                    X_batch_e = X_batch[(Y_hat<self.tau),:]
                    Z_batch_e = Z_batch[(Y_hat<self.tau)]
                    
                    # calculate x + effort
                    X_hat_max = self.effort(self.model, dataset, X_batch_e)
                    
                    # find adversarial model
                    if (alpha > 0):
                        if epoch % 20 == 0:
                            model_adv = self.get_model_adv(X_hat_max, Z_batch_e, curr_alpha, pga_n_epochs)
                    else:
                        model_adv = self.model
                    
                    # get predictions with adversarial theta
                    Y_hat_max = model_adv(X_hat_max).reshape(-1)
                    # calculate fairness loss
                    batch_fair_loss = self.proxy(Z_batch_e, Y_hat_max, self.tau)
                
                batch_loss += lamb*batch_fair_loss

                # update theta
                if torch.isnan(batch_loss).any():
                    continue
                batch_loss.backward()
                optimizer.step()
            
                # track batch metrics
                batch_pred_losses.append(batch_pred_loss.item())
                batch_fair_losses.append(batch_fair_loss.item())
                batch_losses.append(batch_loss.item())
            
            # update scheduler
            scheduler.step()
            
            # track training metrics
            self.train_history.pred_loss.append(np.mean(batch_pred_losses))
            self.train_history.fair_loss.append(np.mean(batch_fair_losses))
            self.train_history.total_loss.append(np.mean(batch_losses))
            self.train_history.theta.append(deepcopy(self.model.get_theta()))
            self.train_history.theta_adv.append(deepcopy(model_adv.get_theta()))
            
            # check for convergence (early termination)
            loss_diff = torch.abs(prev_loss - torch.mean(torch.tensor(batch_losses)))
            if loss_diff < abstol:
                break
                
            prev_loss = torch.mean(torch.tensor(batch_losses))
        
        return self
        
    def predict(self,
                dataset: FairnessDataset,
                alpha: float,
                pga_n_epochs: int = 50
                ):
        
        loss_fn = torch.nn.BCELoss(reduction='mean')

        # get predictions with theta
        Y_hat = self.model(dataset.X).reshape(-1)
        
        # calculate prediction loss
        pred_loss =  loss_fn(Y_hat, dataset.Y)
        
        # if everyone recieved positive label from theta 
        if torch.sum(Y_hat<self.tau) == 0:
            X_hat_max = dataset.X[(Y_hat<self.tau).reshape(-1),:]
            # set fairness loss to 0
            fair_loss = torch.tensor([0.]).float()
            # set adversarial model as the original model 
            model_adv = deepcopy(self.model)
        else:
            # instances that received a negative label from theta
            X_e = dataset.X[(Y_hat<self.tau).reshape(-1),:]
            Z_e = dataset.Z[(Y_hat<self.tau)]
            
            # calculate x + effort
            X_hat_max = self.effort(self.model, dataset, X_e)
            
            # find adversarial model
            if alpha > 0:
                model_adv = self.get_model_adv(X_hat_max, Z_e, alpha, pga_n_epochs)
            else:
                model_adv = self.model
            
            # get predictions with adversarial theta
            Y_hat_max = model_adv(X_hat_max).reshape(-1)
            # calculate fairness loss
            fair_loss = self.proxy(Z_e, Y_hat_max, self.tau)    
            
        
        Y_hat = Y_hat.detach().float().numpy()
        Y_hat_max = model_adv(X_hat_max).reshape(-1).detach().float().numpy()
        pred_loss = pred_loss.detach().item()
        fair_loss = fair_loss.detach().item()
        
        self.model_adv = model_adv
        
        return Y_hat, Y_hat_max, pred_loss, fair_loss
    