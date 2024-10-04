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
    proxy_value = torch.square(torch.mean((Z-Z.mean())*Y_hat_max))
    return proxy_value

def fair_batch_proxy(Z: torch.tensor, Y_hat_max: torch.tensor):
    proxy_value = torch.tensor(0.)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    loss_mean = loss_fn(Y_hat_max, torch.ones(len(Y_hat_max)))
    for z in [0,1]:
        z = int(z)
        group_idx = (Z==z)
        if group_idx.sum() == 0:
            loss_z = torch.tensor(0.)
        else:
            loss_z = loss_fn(Y_hat_max[group_idx], torch.ones(group_idx.sum()))
        proxy_value += torch.abs(loss_z - loss_mean)
    return proxy_value

def covariance_proxy(Z: torch.tensor, Y_hat_max: torch.tensor):
    proxy_value = torch.square(torch.mean((Z-Z.mean())*Y_hat_max))
    return proxy_value



class EIModel():
    def __init__(self, model: LR | NN, proxy: Callable, effort: Effort, model_adv: LR | NN | None = None, tau: float = 0.5) -> None:
        self.model = model
        self.model_adv = model_adv
        self.proxy = proxy
        self.effort = effort
        self.tau = tau
        self.train_history = SimpleNamespace()
        
    def train(self, 
              dataset: FairnessDataset, 
              lamb: float,
              alpha: float = 0.,
              lr: float = 1e-3,
              n_epochs: int = 100,
              batch_size: int = 1024,
              abstol: float = 1e-7,
              pga_n_iters: int = 50 
              ):

        generator = torch.Generator().manual_seed(0)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        
        pred_losses = []
        fair_losses = []
        total_losses = []
        theta_advs = []
        thetas = []
        
        
        loss_fn = torch.nn.BCELoss(reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # THIS IS FOR CONVERGENCE
        loss_diff = 1.
        prev_loss = torch.tensor(0.)
        
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
                    X_batch_e = X_batch[(Y_hat<self.tau), :]
                    Z_batch_e = Z_batch[(Y_hat<self.tau)]
                    
                    X_hat_max = self.effort(self.model, dataset, X_batch_e)

                    # Projected Gradient Ascent   
                    if alpha > 0:
                        if self.model_adv is not None:
                            model_adv =  self.model_adv
                        else:
                            for module in self.model.layers:
                                if hasattr(module, 'weight'):
                                    weight_min = module.weight.data - curr_alpha
                                    weight_max = module.weight.data + curr_alpha
                                if hasattr(module, 'bias'):
                                    bias_min = module.bias.data - curr_alpha
                                    bias_max = module.bias.data + curr_alpha
                            
                            model_adv = deepcopy(self.model).xavier_init().clamp((weight_min, weight_max), (bias_min, bias_max))
                            optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-3, maximize=True)
                                    
                            loss_diff_pga = 1.
                            fair_loss_pga = torch.tensor(0.)

                            for _ in range(int(pga_n_iters)):
                                prev_loss = fair_loss_pga.clone().detach()
                                
                                Y_hat_max_pga = model_adv(X_hat_max).reshape(-1)
                                
                                fair_loss_pga = torch.tensor(0.)
                                loss_fn = torch.nn.MSELoss(reduction='mean')

                                loss_mean = loss_fn(Y_hat_max_pga, torch.ones(len(Y_hat_max_pga)))
                                for z in [0,1]:
                                    z = int(z)
                                    group_idx = (Z_batch_e==z)
                                    if group_idx.sum() == 0:
                                        loss_z = torch.tensor(0.)
                                    else:
                                        loss_z = loss_fn(Y_hat_max_pga[group_idx], torch.ones(group_idx.sum()))
                                    fair_loss_pga += torch.abs(loss_z - loss_mean)
                
                                optimizer_adv.zero_grad()
                                fair_loss_pga.backward()
                                optimizer_adv.step()
                                
                                for module in model_adv.layers:
                                    if hasattr(module, 'weight'):
                                        module.weight.data = module.weight.data.clamp(weight_min, weight_max)
                                    if hasattr(module, 'bias'):
                                        module.bias.data = module.bias.data.clamp(bias_min, bias_max)
                                
                                loss_diff_pga = (prev_loss - fair_loss_pga).abs()
                                # if loss_diff_pga < abstol:
                                #     break
                        
                        Y_hat_max = model_adv(X_hat_max).reshape(-1)
                    else:
                        Y_hat_max = self.model(X_hat_max).reshape(-1)
                    
                    batch_fair_loss = torch.tensor(0.)
                    loss_fn = torch.nn.MSELoss(reduction='mean')
                    loss_mean = loss_fn(Y_hat_max, torch.ones(len(Y_hat_max)))
                    for z in [0,1]:
                        z = int(z)
                        group_idx = (Z_batch_e==z)
                        if group_idx.sum() == 0:
                            loss_z = torch.tensor(0.)
                        else:
                            loss_z = loss_fn(Y_hat_max[group_idx], torch.ones(group_idx.sum()))
                        batch_fair_loss += torch.abs(loss_z - loss_mean)
                
                batch_loss += lamb*batch_fair_loss
                
                optimizer.zero_grad()
                if torch.isnan(batch_loss).any():
                    continue
                batch_loss.backward()
                optimizer.step()
                
                batch_pred_losses.append(batch_pred_loss.item())
                # if hasattr(batch_fair_loss, 'item'):
                batch_fair_losses.append(batch_fair_loss.item())
                # else:
                    # batch_fair_losses.append(batch_fair_loss)
                batch_losses.append(batch_loss.item())
            
            pred_losses.append(np.mean(batch_pred_losses))
            fair_losses.append(np.mean(batch_fair_losses))
            total_losses.append(np.mean(batch_losses))
            thetas.append(deepcopy(self.model.get_theta()))
            
            # THIS IS FOR CONVERGENCE (I'm not sure if this is correct)
            loss_diff = torch.abs(prev_loss - torch.mean(torch.tensor(batch_losses)))
            
            if loss_diff < abstol:
                self.train_history.pred_loss = pred_losses
                self.train_history.fair_loss = fair_losses
                self.train_history.total_loss = total_losses
                return self
            
            prev_loss = torch.mean(torch.tensor(batch_losses)).clone().detach()

        self.train_history.pred_loss = pred_losses
        self.train_history.fair_loss = fair_losses
        self.train_history.total_loss = total_losses
        return self
        
        
    def predict(self,
                dataset: FairnessDataset,
                alpha: float,
                abstol: float = 1e-7,
                pga_n_iters: int = 50
                ):
        
        loss_fn = torch.nn.BCELoss(reduction='mean')
    
        Y_hat = self.model(dataset.X).reshape(-1)
        pred_loss =  loss_fn(Y_hat, dataset.Y)
        
        if torch.sum(Y_hat<self.tau) > 0:
            X_e = dataset.X[(Y_hat<self.tau).reshape(-1),:]
            Z_e = dataset.Z[(Y_hat<self.tau)]
            
            X_hat_max = self.effort(self.model, dataset, X_e)
            if self.model_adv is not None:
                model_adv = self.model_adv
                Y_hat_max = model_adv(X_hat_max).reshape(-1)
                fair_loss = self.proxy(Z_e, Y_hat_max)
            else:
                for module in self.model.layers:
                    if hasattr(module, 'weight'):
                        weight_min = module.weight.data - alpha
                        weight_max = module.weight.data + alpha
                    if hasattr(module, 'bias'):
                        bias_min = module.bias.data.item() - alpha
                        bias_max = module.bias.data.item() + alpha
                
                model_adv = deepcopy(self.model).randn_init().clamp((weight_min, weight_max), (bias_min, bias_max))
                optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-3, maximize=True)
                
                loss_diff = 1.
                # fair_loss = torch.tensor(0.).float()
                for _ in range(pga_n_iters):
                    # prev_loss = fair_loss.clone().detach()
                    Y_hat_max = model_adv(X_hat_max).reshape(-1)

                    fair_loss = torch.tensor(0.)
                    loss_fn = torch.nn.MSELoss(reduction='mean')
                    loss_mean = loss_fn(Y_hat_max, torch.ones(len(Y_hat_max)))
                    for z in [0,1]:
                        z = int(z)
                        group_idx = (Z_e==z)
                        if group_idx.sum() == 0:
                            loss_z = torch.tensor(0.)
                        else:
                            loss_z = loss_fn(Y_hat_max[group_idx], torch.ones(group_idx.sum()))
                        fair_loss += torch.abs(loss_z - loss_mean)
                    
                    optimizer_adv.zero_grad()
                    fair_loss.backward()
                    optimizer_adv.step()
                    
                    # loss_diff = (prev_loss - fair_loss).abs()
                    
                    for module in model_adv.layers:
                        if hasattr(module, 'weight'):
                            module.weight.data = module.weight.data.clamp(weight_min, weight_max)
                        if hasattr(module, 'bias'):
                            module.bias.data = module.bias.data.clamp(bias_min, bias_max)

                    # if loss_diff < abstol:
                    #     break
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
    

class EIModel_S():
    def __init__(self, model: LR | NN, proxy: Callable, effort: Effort, model_adv: LR | NN | None = None, tau: float = 0.5) -> None:
        self.model = model
        self.model_adv = model_adv
        self.proxy = proxy
        self.effort = effort
        self.tau = tau
        self.train_history = SimpleNamespace()
        
    def get_model_adv(self, X_hat_max, Z_e, alpha):
        
        model_adv = deepcopy(self.model)
        theta = model_adv.get_theta()
        X_hat_max = torch.cat((X_hat_max, torch.ones((len(X_hat_max),1))), 1)
        
        A_eq = np.empty((0, len(theta)), float)
        b_eq = np.array([])

        theta = theta.reshape(1,-1).T
        theta.requires_grad = True 
        
        # Y_hat_max = model_adv(X_hat_max).reshape(-1)
        Y_hat_max = torch.nn.Sigmoid()(torch.matmul(X_hat_max, theta)).reshape(-1)
        
        fair_loss = self.proxy(Z_e, Y_hat_max)
        
        gradient_w_loss = grad(fair_loss, theta)[0].reshape(-1)
        # print(A_eq)
        # print(gradient_w_loss)

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
        
    def train(self, 
              dataset: FairnessDataset, 
              lamb: float,
              alpha: float = 0.,
              lr: float = 1e-3,
              n_epochs: int = 100,
              batch_size: int = 1024,
              abstol: float = 1e-7,
              pga_n_iters: int = 50 
              ):

        lamb = torch.tensor(lamb).float()
        generator = torch.Generator().manual_seed(0)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        
        pred_losses = []
        fair_losses = []
        total_losses = []
        thetas = []
        theta_advs = []
        
        loss_fn = torch.nn.BCELoss(reduction='mean')
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # THIS IS FOR CONVERGENCE
        loss_diff = 1.
        prev_loss = torch.tensor(0.)
        
        for epoch in tqdm.trange(n_epochs, desc=f"Training [alpha={alpha:.3f}; lambda={lamb:.5f}; delta={self.effort.delta:.3f}]", unit="epochs", colour='#0091ff'):
            
            batch_losses = []
            batch_pred_losses = []
            batch_fair_losses = []

            curr_alpha = alpha# * (epoch / n_epochs)
            for _, (X_batch, Y_batch, Z_batch) in enumerate(data_loader):
                Y_hat = self.model(X_batch).reshape(-1)
                
                batch_pred_loss = loss_fn(Y_hat, Y_batch)
                batch_loss = (1-lamb)*batch_pred_loss
                
                batch_fair_loss = 0
                if torch.sum(Y_hat<self.tau) > 0:
                    X_batch_e = X_batch[(Y_hat<self.tau), :]
                    Z_batch_e = Z_batch[(Y_hat<self.tau)]
                    
                    X_hat_max = self.effort(self.model, dataset, X_batch_e)

                    # Projected Gradient Ascent   
                    if alpha > 0:
                        if self.model_adv:
                            model_adv = self.model_adv
                        else:
                            model_adv = self.get_model_adv(X_hat_max, Z_batch_e, curr_alpha)
                        Y_hat_max = model_adv(X_hat_max).reshape(-1)
                    else:
                        Y_hat_max = self.model(X_hat_max).reshape(-1)
                    
                    batch_fair_loss = torch.tensor(0.)
                    loss_fn = torch.nn.MSELoss(reduction='mean')
                    loss_mean = loss_fn(Y_hat_max, torch.ones(len(Y_hat_max)))
                    for z in [0,1]:
                        z = int(z)
                        group_idx = (Z_batch_e==z)
                        if group_idx.sum() == 0:
                            loss_z = torch.tensor(0.)
                        else:
                            loss_z = loss_fn(Y_hat_max[group_idx], torch.ones(group_idx.sum()))
                        batch_fair_loss += torch.abs(loss_z - loss_mean)
                else:
                    model_adv = deepcopy(self.model)
                batch_loss += lamb*batch_fair_loss
                
                optimizer.zero_grad()
                if torch.isnan(batch_loss).any():
                    continue
                # if alpha > 0:
                #     batch_fair_loss.backward()
                # else:
                batch_loss.backward()
                optimizer.step()
                
                batch_pred_losses.append(batch_pred_loss.item())
                if hasattr(batch_fair_loss, 'item'):
                    batch_fair_losses.append(batch_fair_loss.item())
                else:
                    batch_fair_losses.append(batch_fair_loss)
                batch_losses.append(batch_loss.item())
            
            pred_losses.append(np.mean(batch_pred_losses))
            fair_losses.append(np.mean(batch_fair_losses))
            total_losses.append(np.mean(batch_losses))
            thetas.append(deepcopy(self.model.get_theta()))
            if alpha > 0:
                theta_advs.append(deepcopy(model_adv.get_theta()))
            
            # THIS IS FOR CONVERGENCE (I'm not sure if this is correct)
            loss_diff = torch.abs(prev_loss - torch.mean(torch.tensor(batch_losses)))
            
            if loss_diff < abstol:
                break
            
            prev_loss = torch.mean(torch.tensor(batch_losses)).clone().detach()

        self.train_history.pred_loss = pred_losses
        self.train_history.fair_loss = fair_losses
        self.train_history.total_loss = total_losses
        self.train_history.thetas = thetas
        if alpha > 0:
            self.train_history.theta_advs = theta_advs
        return self
        
        
    def predict(self,
                dataset: FairnessDataset,
                alpha: float,
                abstol: float = 1e-7,
                pga_n_iters: int = 50
                ):
        
        loss_fn = torch.nn.BCELoss(reduction='mean')
    
        Y_hat = self.model(dataset.X).reshape(-1)
        pred_loss =  loss_fn(Y_hat, dataset.Y)
        
        if torch.sum(Y_hat<self.tau) > 0:
            X_e = dataset.X[(Y_hat<self.tau).reshape(-1),:]
            Z_e = dataset.Z[(Y_hat<self.tau)]
            
            X_hat_max = self.effort(self.model, dataset, X_e) 
            if self.model_adv:
                model_adv = self.model_adv
            else:
                model_adv = self.get_model_adv(X_hat_max, Z_e, alpha)
            Y_hat_max = model_adv(X_hat_max).reshape(-1)

            fair_loss = torch.tensor(0.)
            loss_fn = torch.nn.MSELoss(reduction='mean')
            loss_mean = loss_fn(Y_hat_max, torch.ones(len(Y_hat_max)))
            for z in [0,1]:
                z = int(z)
                group_idx = (Z_e==z)
                if group_idx.sum() == 0:
                    loss_z = torch.tensor(0.)
                else:
                    loss_z = loss_fn(Y_hat_max[group_idx], torch.ones(group_idx.sum()))
                fair_loss += torch.abs(loss_z - loss_mean)
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