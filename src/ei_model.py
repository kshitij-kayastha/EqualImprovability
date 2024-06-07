import tqdm
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from types import SimpleNamespace
from torch.utils.data import DataLoader, random_split
from copy import deepcopy
from abc import ABC, abstractmethod

from data import FairnessDataset
from ei_effort import Effort
from ei_utils import model_performance


class EIModel(ABC):
    def __init__(self, model: nn.Module) -> None:
        self.model = deepcopy(model)
        self.results = SimpleNamespace()
        self.train_history = SimpleNamespace()
        
    @abstractmethod
    def train(self):
        pass
    
    def get_model(self):
        return self.model

    def get_results(self):
        return self.results
        

class FairBatch(EIModel):
    def __init__(self, model, effort_model: Effort, pga_term: int | float = 20, tau: float = 0.5) -> None:
        super(FairBatch, self).__init__(model)
        self.effort_model = effort_model # effort model
        self.model_adv = None
        self.model_adv_r = None
        self.pga_term = pga_term 
        self.tau = tau # threshold
        
    def train(self, 
              dataset: FairnessDataset, 
              lamb: float, 
              alpha: float = 0.,
              lr : float = 1e-2, 
              n_epochs: int = 100, 
              batch_size: int = 1024, 
              ):
        
        generator = torch.Generator().manual_seed(0)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
     
        loss_fn = torch.nn.BCELoss(reduction = 'mean')
        
        pred_losses = [] # total prediction loss
        fair_losses = [] # fairness loss
        
        accuracies = []
        ei_disparities = []
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        for epoch in tqdm.trange(n_epochs, desc=f"Training [alpha={alpha:.2f}; lambda={lamb:.2f}; delta={self.effort_model.delta:.2f}]", unit="epochs", colour='#0091ff'):
        
            batch_pred_losses = [] # batch prediction loss
            batch_fair_losses = [] # batch fairness loss

            for _, (X_batch, Y_batch, Z_batch) in enumerate(train_loader):
                Y_hat = self.model(X_batch)
                
                batch_loss = 0

                # prediction loss
                batch_pred_loss = loss_fn(Y_hat.reshape(-1), Y_batch)
                batch_loss += (1-lamb)*batch_pred_loss

                # fairness loss
                batch_fair_loss = 0
                # EI_Constraint
                if torch.sum(Y_hat<self.tau) > 0:
                    X_batch_e = X_batch[(Y_hat<self.tau).reshape(-1),:]
                    Z_batch_e = Z_batch[(Y_hat<self.tau).reshape(-1)]

                    # PGA
                    if alpha > 0: 
                        model_adv = deepcopy(self.model)
                        optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-2, maximize=True)
                        pga_loss_fn = torch.nn.BCELoss(reduction = 'mean')
                        
                        for module in model_adv.layers:
                            if hasattr(module, 'weight'):
                                weight_min = module.weight.data - alpha
                                weight_max = module.weight.data + alpha
                            if hasattr(module, 'bias'):
                                bias_min = module.bias.data.item() - alpha
                                bias_max = module.bias.data.item() + alpha
                        
                        for _ in range(self.pga_term):
                            pga_fair_loss = 0.
                            # Effort delta 
                            X_hat_pga = self.effort_model(model_adv, dataset, X_batch_e)
                            Y_hat_pga = model_adv(X_hat_pga)
                            pga_loss_mean = pga_loss_fn(Y_hat_pga.reshape(-1), torch.ones(len(Y_hat_pga)))
                            pga_loss_z = torch.zeros(len(dataset.sensitive_attrs))
                            for z in dataset.sensitive_attrs:
                                z = int(z)
                                group_idx = (Z_batch_e == z)
                                if group_idx.sum() == 0:
                                    continue
                                pga_loss_z[z] = pga_loss_fn(Y_hat_pga.reshape(-1)[group_idx], torch.ones(group_idx.sum()))
                                pga_fair_loss += torch.abs(pga_loss_z[z] - pga_loss_mean)
                            
                            optimizer_adv.zero_grad()
                            pga_fair_loss.backward()
                            optimizer_adv.step()
                            
                            for module in model_adv.layers:
                                if hasattr(module, 'weight'):
                                    with torch.no_grad():
                                        module.weight.data = module.weight.data.clamp(weight_min, weight_max)
                                if hasattr(module, 'bias'):
                                    with torch.no_grad():
                                        module.bias.data = module.bias.data.clamp(bias_min, bias_max)

                        Y_hat_max = Y_hat_pga.clone().detach().requires_grad_()
                    else:
                        X_hat_max = self.effort_model(self.model, dataset, X_batch_e)
                        Y_hat_max = self.model(X_hat_max)
                    
                    loss_mean = loss_fn(Y_hat_max.reshape(-1), torch.ones(len(Y_hat_max)))
                    loss_z = torch.zeros(len(dataset.sensitive_attrs))
                    for z in dataset.sensitive_attrs:
                        z = int(z)
                        group_idx = (Z_batch_e == z)
                        if group_idx.sum() == 0:
                            continue
                        loss_z[z] = loss_fn(Y_hat_max.reshape(-1)[group_idx], torch.ones(group_idx.sum()))
                        batch_fair_loss += torch.abs(loss_z[z] - loss_mean)

                batch_loss += lamb*batch_fair_loss
                
                optimizer.zero_grad()
                if (torch.isnan(batch_loss)).any():
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
            X_hat_max_train = self.effort_model(self.model, dataset, dataset.X)
            Y_hat_max_train = self.model(X_hat_max_train)
            Y_hat_max_train = Y_hat_max_train.reshape(-1).detach().numpy()
            
            accuracy, ei_disparity = model_performance(dataset.Y.detach().numpy(), dataset.Z.detach().numpy(), Y_hat_train, Y_hat_max_train, self.tau)
            
            accuracies.append(accuracy)
            ei_disparities.append(ei_disparity)

        self.train_history.accuracy = accuracies
        self.train_history.p_loss = pred_losses
        self.train_history.f_loss = fair_losses
        self.train_history.ei_disparity = ei_disparities
        return self
    
    def predict_r(self, dataset, alpha):
        Y_hat = self.model(dataset.X).reshape(-1).detach().numpy()
        model_adv = deepcopy(self.model)
        
        if alpha > 0: 
            optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-2, maximize=True)
            pga_loss_fn = torch.nn.BCELoss(reduction = 'mean')
            
            for module in model_adv.layers:
                if hasattr(module, 'weight'):
                    weight_min = module.weight.data - alpha
                    weight_max = module.weight.data + alpha
                if hasattr(module, 'bias'):
                    bias_min = module.bias.data.item() - alpha
                    bias_max = module.bias.data.item() + alpha
            
            for _ in range(self.pga_term):
                X_hat_pga = self.effort_model(model_adv, dataset, dataset.X)
                Y_hat_pga = model_adv(X_hat_pga)
                
                pga_fair_loss = 0.
                pga_loss_mean = pga_loss_fn(Y_hat_pga.reshape(-1), torch.ones(len(Y_hat_pga)))
                pga_loss_z = torch.zeros(len(dataset.sensitive_attrs))
                for z in dataset.sensitive_attrs:
                    z = int(z)
                    group_idx = (dataset.Z == z)
                    if group_idx.sum() == 0:
                        continue
                    pga_loss_z[z] = pga_loss_fn(Y_hat_pga.reshape(-1)[group_idx], torch.ones(group_idx.sum()))
                    pga_fair_loss += torch.abs(pga_loss_z[z] - pga_loss_mean)
                
                optimizer_adv.zero_grad()
                pga_fair_loss.backward()
                optimizer_adv.step()
                
                for module in model_adv.layers:
                    if hasattr(module, 'weight'):
                        with torch.no_grad():
                            module.weight.data = module.weight.data.clamp(weight_min, weight_max)
                    if hasattr(module, 'bias'):
                        with torch.no_grad():
                            module.bias.data = module.bias.data.clamp(bias_min, bias_max)

            Y_hat_max = Y_hat_pga.clone().detach().requires_grad_()
        else:
            X_hat_max = self.effort_model(self.model, dataset, dataset.X)
            Y_hat_max = self.model(X_hat_max)
        
        self.model_adv_r = model_adv
        
        return Y_hat, Y_hat_max.reshape(-1).detach().numpy()
    
    def predict(self, dataset, alpha):
        Y_hat = self.model(dataset.X).reshape(-1).detach().numpy()
        
        X_hat_max = self.effort_model(self.model, dataset, dataset.X)
        model_adv = deepcopy(self.model)
        if alpha > 0: 
            optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-2, maximize=True)
            pga_loss_fn = torch.nn.BCELoss(reduction = 'mean')
            
            for module in model_adv.layers:
                if hasattr(module, 'weight'):
                    weight_min = module.weight.data - alpha
                    weight_max = module.weight.data + alpha
                if hasattr(module, 'bias'):
                    bias_min = module.bias.data.item() - alpha
                    bias_max = module.bias.data.item() + alpha
            
            for _ in range(self.pga_term):
                Y_hat_pga = model_adv(X_hat_max)
                
                pga_fair_loss = 0.
                pga_loss_mean = pga_loss_fn(Y_hat_pga.reshape(-1), torch.ones(len(Y_hat_pga)))
                pga_loss_z = torch.zeros(len(dataset.sensitive_attrs))
                for z in dataset.sensitive_attrs:
                    z = int(z)
                    group_idx = (dataset.Z == z)
                    if group_idx.sum() == 0:
                        continue
                    pga_loss_z[z] = pga_loss_fn(Y_hat_pga.reshape(-1)[group_idx], torch.ones(group_idx.sum()))
                    pga_fair_loss += torch.abs(pga_loss_z[z] - pga_loss_mean)
                
                optimizer_adv.zero_grad()
                pga_fair_loss.backward()
                optimizer_adv.step()
                
                for module in model_adv.layers:
                    if hasattr(module, 'weight'):
                        with torch.no_grad():
                            module.weight.data = module.weight.data.clamp(weight_min, weight_max)
                    if hasattr(module, 'bias'):
                        with torch.no_grad():
                            module.bias.data = module.bias.data.clamp(bias_min, bias_max)

            Y_hat_max = model_adv(X_hat_max)
        else:
            Y_hat_max = self.model(X_hat_max)
        
        self.model_adv = model_adv
        
        return Y_hat, Y_hat_max.reshape(-1).detach().numpy()
    
    def evaluate(self, dataset, alpha):
        
        Y_hat, Y_hat_max = self.predict(dataset, alpha)
        accuracy, ei_disparity = model_performance(dataset.Y.detach().numpy(), dataset.Z.detach().numpy(), Y_hat, Y_hat_max, self.tau)
        
        return accuracy, ei_disparity
    
    
    
    
class Covariance(EIModel):
    def __init__(self, model, effort_model: Effort, pga_iter: int = 20, tau: float = 0.5) -> None:
        super(Covariance, self).__init__(model)
        self.effort_model = effort_model # effort model
        self.pga_iter = pga_iter
        self.tau = tau # threshold
        
    def train(self, 
              dataset: FairnessDataset, 
              lamb: float, 
              alpha: float,
              lr=1e-2, 
              n_epochs=100, 
              batch_size=1024, 
              ):
        generator = torch.Generator().manual_seed(0)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
     
        loss_fn = torch.nn.BCELoss(reduction = 'mean')
        
        pred_losses = [] # total prediction loss
        fair_losses = [] # fairness loss
        
        accuracies = []
        ei_disparities = []
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        for epoch in tqdm.trange(n_epochs, desc=f"Training [alpha={alpha:.4f}; lambda={lamb:.4f}; delta={self.effort_model.delta:.4f}]", unit="epochs", colour='#0091ff'):
        
            batch_pred_losses = [] # batch prediction loss
            batch_fair_losses = [] # batch fairness loss

            for _, (X_batch, Y_batch, Z_batch) in enumerate(train_loader):
                Y_hat = self.model(X_batch)
                
                batch_loss = 0

                # prediction loss
                batch_pred_loss = loss_fn(Y_hat.reshape(-1), Y_batch)
                batch_loss += (1-lamb)*batch_pred_loss

                batch_fair_loss = 0
                # EI_Constraint
                if torch.sum(Y_hat<self.tau) > 0:
                    X_batch_e = X_batch[(Y_hat<self.tau).reshape(-1),:]
                    Z_batch_e = Z_batch[(Y_hat<self.tau).reshape(-1)]
                   
                    pga_fair_loss = 0
                    # PGA
                    if alpha > 0: 
                        model_adv = deepcopy(self.model)
                        optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-2, maximize=True)
                        
                        for module in model_adv.layers:
                            if hasattr(module, 'weight'):
                                weight_min = module.weight.data - alpha
                                weight_max = module.weight.data + alpha
                            if hasattr(module, 'bias'):
                                bias_min = module.bias.data.item() - alpha
                                bias_max = module.bias.data.item() + alpha
                        
                        for _ in range(self.pga_iter):
                            pga_fair_loss = 0
                            # Effort delta 
                            Y_hat_pga = self.effort_model(model_adv, dataset, X_batch_e)
                            
                            pga_fair_loss = torch.square(torch.mean((Z_batch_e-Z_batch_e.mean()) * Y_hat_pga.reshape(-1)))
                            
                            optimizer_adv.zero_grad()
                            pga_fair_loss.backward()
                            optimizer_adv.step()
                            
                            for module in model_adv.layers:
                                if hasattr(module, 'weight'):
                                    with torch.no_grad():
                                        module.weight.data = module.weight.data.clamp(weight_min, weight_max)
                                if hasattr(module, 'bias'):
                                    with torch.no_grad():
                                        module.bias.data = module.bias.data.clamp(bias_min, bias_max)

                        Y_hat_max = Y_hat_pga.clone().detach().requires_grad_()
                    else:
                        Y_hat_max = self.effort_model(self.model, dataset, X_batch_e)
                    
                    batch_fair_loss += torch.square(torch.mean((Z_batch_e-Z_batch_e.mean())*Y_hat_max.reshape(-1)))

                batch_loss += lamb*batch_fair_loss
                
                optimizer.zero_grad()
                if (torch.isnan(batch_loss)).any():
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
            Y_hat_max_train = self.effort_model(self.model, dataset, dataset.X)
            Y_hat_max_train = Y_hat_max_train.reshape(-1).detach().numpy()
            
            accuracy, ei_disparity = model_performance(dataset.Y.detach().numpy(), dataset.Z.detach().numpy(), Y_hat_train, Y_hat_max_train, self.tau)
            
            accuracies.append(accuracy)
            ei_disparities.append(ei_disparity)

        self.train_history.accuracy = accuracies
        self.train_history.p_loss = pred_losses
        self.train_history.f_loss = fair_losses
        self.train_history.ei_disparity = ei_disparities
        return self
    
    def predict(self, dataset, alpha):
        Y_hat = self.model(dataset.X).reshape(-1).detach().numpy()
        model_adv = deepcopy(self.model)
        if alpha > 0: 
            optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-2, maximize=True)
            
            for module in model_adv.layers:
                if hasattr(module, 'weight'):
                    weight_min = module.weight.data - alpha
                    weight_max = module.weight.data + alpha
                if hasattr(module, 'bias'):
                    bias_min = module.bias.data.item() - alpha
                    bias_max = module.bias.data.item() + alpha
            
            for _ in range(self.pga_iter):
                Y_hat_pga = self.effort_model(model_adv, dataset, dataset.X)
                
                pga_fair_loss = torch.square(torch.mean((dataset.Z-dataset.Z.mean()) * Y_hat_pga.reshape(-1)))
                
                optimizer_adv.zero_grad()
                pga_fair_loss.backward()
                optimizer_adv.step()
                
                for module in model_adv.layers:
                    if hasattr(module, 'weight'):
                        with torch.no_grad():
                            module.weight.data = module.weight.data.clamp(weight_min, weight_max)
                    if hasattr(module, 'bias'):
                        with torch.no_grad():
                            module.bias.data = module.bias.data.clamp(bias_min, bias_max)

            Y_hat_max = Y_hat_pga.clone().detach()
        else:
            Y_hat_max = self.effort_model(self.model, dataset, dataset.X)
        
        return Y_hat, Y_hat_max.reshape(-1).detach().numpy()
    
    def evaluate(self, dataset, alpha):
        
        Y_hat, Y_hat_max = self.predict(dataset, alpha)
        accuracy, ei_disparity = model_performance(dataset.Y.detach().numpy(), dataset.Z.detach().numpy(), Y_hat, Y_hat_max, self.tau)
        
        return accuracy, ei_disparity