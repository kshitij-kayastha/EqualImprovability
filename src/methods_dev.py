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
from torch.utils.data import DataLoader, random_split

from src.model import LR, NN
from src.effort import Effort
from src.data import FairnessDataset
from src.utils import model_performance


def covariance_proxy(Z: torch.tensor, Y_hat_max: torch.tensor):
    fair_loss = torch.square(torch.mean((Z-Z.mean())*Y_hat_max))
    return fair_loss

def fair_batch_proxy(Z: torch.tensor, Y_hat_max: torch.tensor):
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
    def __init__(self, model: LR | NN, proxy: Callable, effort: Effort, tau: float = 0.5, pga: bool = True) -> None:
        self.model = model
        self.proxy = proxy
        self.effort = effort
        self.tau = tau
        self.pga = pga
        self.train_history = SimpleNamespace()
        
    def get_model_adv_solver(self, X_hat_max, Z, alpha):
        
        model_adv = deepcopy(self.model)
        theta = model_adv.get_theta()
        X_hat_max = torch.cat((X_hat_max, torch.ones((len(X_hat_max),1))), 1)
        
        A_eq = np.empty((0, len(theta)), float)
        b_eq = np.array([])

        theta = theta.reshape(1,-1).T
        theta.requires_grad = True 
        
        Y_hat_max = torch.nn.Sigmoid()(torch.matmul(X_hat_max, theta)).reshape(-1)
        
        fair_loss = self.proxy(Z, Y_hat_max)
        
        gradient_w_loss = grad(fair_loss, theta)[0].reshape(-1)

        c = list(np.array(gradient_w_loss) * np.array([-1] * len(gradient_w_loss)))
        bound = (-alpha, alpha)
        bounds = [bound] * len(gradient_w_loss)

        res = linprog(c, bounds=bounds, A_eq=A_eq, b_eq=b_eq, method='simplex')
        alpha_opt = res.x  # the delta value that maximizes the function
        weights_alpha, bias_alpha = torch.from_numpy(alpha_opt[:-1]).float(), torch.from_numpy(alpha_opt[[-1]]).float()
        
        for module in model_adv.layers:
            if hasattr(module, 'weight'):
                module.weight.data += weights_alpha.reshape(1,-1)
                
            if hasattr(module, 'bias'):
                module.bias.data += bias_alpha
        
        return model_adv
        
    def get_model_adv_pga(self, X_hat_max: torch.Tensor, Z: torch.Tensor, alpha: float, n_epochs: int, pga_abstol: float):
        for module in self.model.layers:
            if hasattr(module, 'weight'):
                weight_min = module.weight.data - alpha
                weight_max = module.weight.data + alpha
            if hasattr(module, 'bias'):
                bias_min = module.bias.data - alpha
                bias_max = module.bias.data + alpha
        
        model_adv = deepcopy(self.model).xavier_init().clamp((weight_min, weight_max), (bias_min, bias_max))
        optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-3, maximize=True)
                
        loss_diff = 1.
        fair_loss = torch.tensor(0.)
        for _ in range(int(n_epochs)):
            prev_loss = fair_loss.clone().detach()
            
            Y_hat_max = model_adv(X_hat_max).reshape(-1)
            
            fair_loss = torch.tensor(0.)
            loss_fn = torch.nn.MSELoss(reduction='mean')

            loss_mean = loss_fn(Y_hat_max, torch.ones(len(Y_hat_max)))
            for z in [0,1]:
                z = int(z)
                group_idx = (Z==z)
                if group_idx.sum() == 0:
                    loss_z = torch.tensor(0.)
                else:
                    loss_z = loss_fn(Y_hat_max[group_idx], torch.ones(group_idx.sum()))
                fair_loss += torch.abs(loss_z - loss_mean)

            optimizer_adv.zero_grad()
            fair_loss.backward()
            optimizer_adv.step()
            
            model_adv.clamp((weight_min, weight_max), (bias_min, bias_max))
                    
            loss_diff = (prev_loss - fair_loss).abs()
            # if loss_diff < pga_abstol:
            #     break
        return model_adv
        
        
    def train(self, 
              dataset: FairnessDataset, 
              lamb: float,
              alpha: float = 0.,
              lr: float = 1e-3,
              n_epochs: int = 100,
              batch_size: int = 1024,
              abstol: float = 1e-7,
              pga_n_epochs: int = 50,
              pga_abstol: float = 1e-7 
              ):
        
        self.train_history.pred_loss = []
        self.train_history.fair_loss = []
        self.train_history.total_loss = []
        self.train_history.theta_adv = []
        self.train_history.theta = []
        
        lamb = torch.tensor(lamb).float()
        generator = torch.Generator().manual_seed(0)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        
        loss_fn = torch.nn.BCELoss(reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        loss_diff = 1.
        prev_loss = torch.tensor(0.)
        early_stop_epoch = None
        
        for epoch in tqdm.trange(n_epochs, desc=f"Training [alpha={alpha:.3f}; lambda={lamb:.5f}; delta={self.effort.delta:.3f}]", unit="epochs", colour='#0091ff'):
            
            batch_losses = []
            batch_pred_losses = []
            batch_fair_losses = []

            curr_alpha = alpha * (epoch / n_epochs)
            for _, (X_batch, Y_batch, Z_batch) in enumerate(data_loader):
                Y_hat = self.model(X_batch).reshape(-1)
                
                batch_pred_loss = loss_fn(Y_hat, Y_batch)
                batch_loss = (1-lamb)*batch_pred_loss
                
                batch_fair_loss = torch.tensor(0.)
                if torch.sum(Y_hat<self.tau) > 0:
                    optimizer.zero_grad()
                    
                    X_batch_e = X_batch[(Y_hat<self.tau),:]
                    Z_batch_e = Z_batch[(Y_hat<self.tau)]
                    
                    X_hat_max = self.effort(self.model, dataset, X_batch_e)
                    if alpha > 0:
                        if self.pga:
                            model_adv = self.get_model_adv_pga(X_hat_max, Z_batch_e, curr_alpha, pga_n_epochs, pga_abstol)
                        else:
                            model_adv = self.get_model_adv_solver(X_hat_max, Z_batch_e, curr_alpha)
                    else:
                        model_adv = self.model
                    
                    Y_hat_max = model_adv(X_hat_max).reshape(-1)
                    batch_fair_loss = self.proxy(Z_batch_e, Y_hat_max)
                
                batch_loss += lamb*batch_fair_loss

                if torch.isnan(batch_loss).any():
                    continue
                batch_loss.backward()
                optimizer.step()
            
                batch_pred_losses.append(batch_pred_loss.item())
                batch_fair_losses.append(batch_fair_loss.item())
                batch_losses.append(batch_loss.item())
            
            self.train_history.pred_loss.append(np.mean(batch_pred_losses))
            self.train_history.fair_loss.append(np.mean(batch_fair_losses))
            self.train_history.total_loss.append(np.mean(batch_losses))
            self.train_history.theta.append(deepcopy(self.model.get_theta()))
            self.train_history.theta_adv.append(deepcopy(model_adv.get_theta()))
            
            loss_diff = torch.abs(prev_loss - torch.mean(torch.tensor(batch_losses)))
            if loss_diff < abstol:
                break
                
            prev_loss = torch.mean(torch.tensor(batch_losses))
        
        return self
        
        
    def predict(self,
                dataset: FairnessDataset,
                alpha: float,
                pga_abstol: float = 1e-7,
                pga_n_epochs: int = 50
                ):
        
        loss_fn = torch.nn.BCELoss(reduction='mean')
    
        Y_hat = self.model(dataset.X).reshape(-1)
        pred_loss =  loss_fn(Y_hat, dataset.Y)
        
        if torch.sum(Y_hat<self.tau) > 0:
            X_e = dataset.X[(Y_hat<self.tau).reshape(-1),:]
            Z_e = dataset.Z[(Y_hat<self.tau)]
            
            X_hat_max = self.effort(self.model, dataset, X_e)
            if alpha > 0:
                if self.pga:
                    model_adv = self.get_model_adv_pga(X_hat_max, Z_e, alpha, pga_n_epochs, pga_abstol)
                else:
                    model_adv = self.get_model_adv_solver(X_hat_max, Z_e, alpha)
            else:
                model_adv = self.model
            Y_hat_max = model_adv(X_hat_max).reshape(-1)
            fair_loss = self.proxy(Z_e, Y_hat_max)    
        else:
            X_hat_max = dataset.X[(Y_hat<self.tau).reshape(-1),:]
            fair_loss = torch.tensor([0.]).float()
            model_adv = deepcopy(self.model)
        
        Y_hat = Y_hat.detach().float().numpy()
        Y_hat_max = model_adv(X_hat_max).reshape(-1).detach().float().numpy()
        pred_loss = pred_loss.detach().item()
        fair_loss = fair_loss.detach().item()
        
        self.model_adv = model_adv
        
        return Y_hat, Y_hat_max, pred_loss, fair_loss
    