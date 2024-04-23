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
        self.model = model
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
    def __init__(self, model, effort_model: Effort, tau: float = 0.5) -> None:
        super(FairBatch, self).__init__(model)
        self.effort_model = effort_model # effort model
        self.tau = tau # threshold
        
    def train(self, 
              dataset: FairnessDataset, 
              lamb: float, 
              sensitive_attrs, 
              lr=1e-2, 
              n_epochs=100, 
              batch_size=1024, 
              ):

        generator = torch.Generator().manual_seed(0)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
     
        loss_fn = torch.nn.BCELoss(reduction = 'mean')
        
        p_losses = [] # total prediction loss
        f_losses = [] # fairness loss
        
        accuracies = []
        ei_disparities = []
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        for epoch in tqdm.trange(n_epochs, desc=f"Training [lambda={lamb:.2f}]", unit="epochs", colour='#0091ff'):
        
            batch_p_loss = [] # batch prediction loss
            batch_f_loss = [] # batch fairness loss

            for _, (X_batch, Y_batch, Z_batch) in enumerate(train_loader):
                Y_hat = self.model(X_batch)
                
                loss = 0

                # prediction loss
                p_loss = loss_fn(Y_hat.reshape(-1), Y_batch)
                loss += (1-lamb)*p_loss

                # fairness loss
                f_loss = 0
                # EI_Constraint
                if torch.sum(Y_hat<self.tau) > 0:
                    X_batch_e = X_batch[(Y_hat<self.tau).reshape(-1),:]
                    Z_batch_e = Z_batch[(Y_hat<self.tau).reshape(-1)]
    
                    # Effort delta 
                    Y_hat_max = self.effort_model(self.model, dataset, X_batch_e)

                    loss_mean = loss_fn(Y_hat_max.reshape(-1), torch.ones(len(Y_hat_max)))
                    loss_z = torch.zeros(len(sensitive_attrs))
                    for z in sensitive_attrs:
                        z = int(z)
                        group_idx = (Z_batch_e == z)
                        if group_idx.sum() == 0:
                            continue
                        loss_z[z] = loss_fn(Y_hat_max.reshape(-1)[group_idx], torch.ones(group_idx.sum()))
                        f_loss += torch.abs(loss_z[z] - loss_mean)

                loss += lamb*f_loss
                
                optimizer.zero_grad()
                if (torch.isnan(loss)).any():
                    continue
                loss.backward()
                optimizer.step()

                batch_p_loss.append(p_loss.item())
                if hasattr(f_loss,'item'):
                    batch_f_loss.append(f_loss.item())
                else:
                    batch_f_loss.append(f_loss)
            
            # batch ends
            p_losses.append(np.mean(batch_p_loss))
            f_losses.append(np.mean(batch_f_loss))

            Y_hat_train = self.model(dataset.X).reshape(-1).detach().numpy()
            Y_hat_max_train = self.effort_model(self.model, dataset, dataset.X)
            Y_hat_max_train = Y_hat_max_train.reshape(-1).detach().numpy()
            
            accuracy, ei_disparity = model_performance(dataset.Y.detach().numpy(), dataset.Z.detach().numpy(), Y_hat_train, Y_hat_max_train, self.tau)
            
            accuracies.append(accuracy)
            ei_disparities.append(ei_disparity)

        self.train_history.accuracy = accuracies
        self.train_history.p_loss = p_losses
        self.train_history.f_loss = f_losses
        self.train_history.ei_disparity = ei_disparities
        
        
    def evaluate(self, dataset, alpha):
        
        if alpha != 0.:
            model = deepcopy(self.model)
            for module in model.layers:
                if hasattr(module, 'weight'):
                    module.weight.data += alpha
                if hasattr(module, 'bias'):
                    module.bias.data += alpha
        else:
            model = self.model

        Y_hat = self.model(dataset.X).reshape(-1).detach().numpy()
        Y_hat_max = self.effort_model(model, dataset, dataset.X)
        Y_hat_max = Y_hat_max.reshape(-1).detach().numpy()
        accuracy, ei_disparity = model_performance(dataset.Y.detach().numpy(), dataset.Z.detach().numpy(), Y_hat, Y_hat_max, self.tau)
        
        return accuracy, ei_disparity
    
    def eval(self, dataset, alpha, sensitive_attrs):
        Y_hat = self.model(dataset.X).reshape(-1).detach().numpy()

        model_adv = deepcopy(self.model)
        optimizer_adv = optim.Adam(model_adv.parameters(), lr=1e-3, maximize=False)
        loss_fn = torch.nn.BCELoss(reduction = 'mean')
        
        for module in self.model.layers:
            if hasattr(module, 'weight'):
                weight_min = module.weight.data - alpha
                weight_max = module.weight.data + alpha
            if hasattr(module, 'bias'):
                bias_min = module.bias.data.item() - alpha
                bias_max = module.bias.data.item() + alpha
        
        for _ in range(20):
            pga_f_loss = 0
            # Effort delta 
            Y_hat_pga = self.effort_model(model_adv, dataset, dataset.X)
            
            pga_loss_mean = loss_fn(Y_hat_pga.reshape(-1), torch.ones(len(Y_hat_pga)))
            pga_loss_z = torch.zeros(len(sensitive_attrs))
            for z in sensitive_attrs:
                z = int(z)
                group_idx = (dataset.Z == z)
                if group_idx.sum() == 0:
                    continue
                pga_loss_z[z] = loss_fn(Y_hat_pga.reshape(-1)[group_idx], torch.ones(group_idx.sum()))
                pga_f_loss += torch.abs(pga_loss_z[z] - pga_loss_mean)
            
            optimizer_adv.zero_grad()
            pga_f_loss.backward()
            optimizer_adv.step()
            
            for module in model_adv.layers:
                if hasattr(module, 'weight'):
                    # with torch.no_grad():
                    module.weight.data = module.weight.data.clamp(weight_min, weight_max)
                if hasattr(module, 'bias'):
                    # with torch.no_grad():
                    module.bias.data = module.bias.data.clamp(bias_min, bias_max)
        
        Y_hat_max = Y_hat_pga.reshape(-1).detach().numpy()
        accuracy, ei_disparity = model_performance(dataset.Y.detach().numpy(), dataset.Z.detach().numpy(), Y_hat, Y_hat_max, self.tau)
        
        return accuracy, ei_disparity