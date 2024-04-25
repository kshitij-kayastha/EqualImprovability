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
        

class FairBatch(EIModel):
    def __init__(self, model, effort_model: Effort, tau: float = 0.5) -> None:
        super(FairBatch, self).__init__(model)
        self.effort_model = effort_model # effort model
        self.tau = tau # threshold
        
    def train(self, 
              dataset: FairnessDataset, 
              lamb: float, 
              sensitive_attrs, 
              lr=1e-3,
              n_epochs=100, 
              batch_size=1024, 
              ):

        generator = torch.Generator().manual_seed(0)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
     
        loss_fn = torch.nn.BCELoss(reduction = 'mean')
        
        pred_losses = [] 
        fair_losses = [] 
        accuracies = []
        ei_disparities = []
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        for epoch in tqdm.trange(n_epochs, desc=f"Training [lambda={lamb:.2f}]", unit="epochs", colour='#0091ff'):
        
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

            Y_hat = self.model(dataset.X).reshape(-1).detach().numpy()
            Y_hat_max = self.effort_model(self.model, dataset, dataset.X)
            Y_hat_max = Y_hat_max.reshape(-1).detach().numpy()
            
            accuracy, ei_disparity = model_performance(dataset.Y.detach().numpy(), dataset.Z.detach().numpy(), Y_hat, Y_hat_max, self.tau)
            
            accuracies.append(accuracy)
            ei_disparities.append(ei_disparity)

        self.train_history.accuracy = accuracies
        self.train_history.p_loss = pred_losses
        self.train_history.f_loss = fair_losses
        self.train_history.ei_disparity = ei_disparities
        
    def predict(self, dataset):
        Y_hat = self.model(dataset.X).reshape(-1).detach().numpy()
        Y_hat_max = self.effort_model(self.model, dataset, dataset.X)
        
        return Y_hat, Y_hat_max.reshape(-1).detach().numpy()
        
    def evaluate(self, dataset):
        Y_hat, Y_hat_max = self.predict(dataset)
        accuracy, ei_disparity = model_performance(dataset.Y.detach().numpy(), dataset.Z.detach().numpy(), Y_hat, Y_hat_max, self.tau)
        
        return accuracy, ei_disparity