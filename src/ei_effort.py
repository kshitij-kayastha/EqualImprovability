import torch
from torch.autograd import Variable
from abc import ABC, abstractmethod
from data import FairnessDataset

class Effort(ABC):
    def __init__(self, delta) -> None:
        super().__init__()
        self.delta = delta
        
    @abstractmethod
    def __call__(self, model, dataset, x) -> torch.Tensor:
        pass
    
    
class PGD_Effort(Effort):
    def __init__(self, delta, n_epochs=20, lr=1) -> None:
        super().__init__(delta)
        self.n_epochs = n_epochs
        self.lr = lr
         
    def __call__(self, model, dataset: FairnessDataset, x: torch.Tensor) -> torch.Tensor:
        efforts = Variable(torch.zeros(x.shape), requires_grad = True)
        
        improvable_indices = []
        for i in range(efforts.shape[1]):
            if i not in dataset.U_index:
                improvable_indices.append(i)
        
        C_min = torch.zeros(x[:, dataset.C_index].shape)
        C_max = torch.zeros(x[:, dataset.C_index].shape)
        for j in range(len(dataset.C_index)):        
            C_min[:, j] = dataset.C_min[j]-x[:, dataset.C_index[j]]
            C_max[:, j] = dataset.C_max[j]-x[:, dataset.C_index[j]]
        
        loss_fn = torch.nn.BCELoss(reduction='sum')
        for i in range(self.n_epochs):
            y_hat = model(x + efforts)
            cost = loss_fn(y_hat.squeeze(), torch.ones(y_hat.squeeze().shape))
            model.zero_grad()
            cost.backward()

            efforts_update = efforts - (self.lr/((i+1)**.5))*efforts.grad
            efforts_update[:, improvable_indices] = torch.clamp(efforts_update[:, improvable_indices], -self.delta, self.delta)

            efforts_update[:,dataset.C_index] = efforts_update[:, dataset.C_index].round()
            efforts_update[:,dataset.C_index] = efforts_update[:, dataset.C_index].clamp(C_min, C_max)
            efforts_update[:,dataset.U_index] = torch.zeros(efforts[:, dataset.U_index].shape)
            efforts = Variable(efforts_update, requires_grad = True)
            
        y_hat = model(x + efforts)

        return y_hat
    
    
class Optimal_Effort(Effort):
    def __init__(self, delta, norm='inf') -> None:
        super().__init__(delta)
        self.norm = norm
        
    def __call__(self, model, dataset: FairnessDataset, x: torch.Tensor):
        efforts = Variable(torch.zeros(x.shape), requires_grad = True)
    
        improvable_indices = []
        for i in range(efforts.shape[1]):
            if i not in dataset.U_index:
                improvable_indices.append(i)
                
        C_min = torch.zeros(x[:, dataset.C_index].shape)
        C_max = torch.zeros(x[:, dataset.C_index].shape)
        for j in range(len(dataset.C_index)):        
            C_min[:, j] = dataset.C_min[j]-x[:, dataset.C_index[j]]
            C_max[:, j] = dataset.C_max[j]-x[:, dataset.C_index[j]]

        for name, param in model.named_parameters():
            if 'layers.0.weight' in name:
                weights = param.detach() 
        if self.norm=='inf':
            efforts_update = self.delta*torch.sign(weights)*torch.ones(x.shape)
            efforts_update[:, improvable_indices] = efforts_update[:, improvable_indices].clamp(-self.delta, self.delta)
        elif self.norm=='l2':
            efforts_update = self.delta*(weights / torch.square(torch.sum(weights*weights)))*torch.ones(x.shape)
        efforts_update[:, dataset.C_index] = efforts_update[:, dataset.C_index].round()
        efforts_update[:, dataset.C_index] = efforts_update[:, dataset.C_index].clamp(C_min, C_max)
        efforts_update[:, dataset.U_index] = torch.zeros(efforts[:, dataset.U_index].shape)

        efforts = Variable(efforts_update, requires_grad = True)
            
        y_hat = model(x + efforts)

        return y_hat