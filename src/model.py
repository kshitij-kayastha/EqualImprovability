import torch
import torch.nn as nn
import numpy as np

class LR(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_features,1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.layers(x)
        return x
    
    def set_theta(self, theta):
        if isinstance(theta, np.ndarray):
            theta = torch.from_numpy(theta).float()
        for module in self.layers:
            if hasattr(module, 'weight'):
                module.weight.data = theta[:-1].reshape(1,-1)
            if hasattr(module, 'bias'):
                module.bias.data = theta[[-1]]
        return self
    
    def get_theta(self):
        for module in self.layers:
            if hasattr(module, 'weight'):
                weights = module.weight.data
            if hasattr(module, 'bias'):
                bias = module.bias.data
        theta = torch.cat((weights[0], bias), 0)
        return theta.clone().detach()

    def xavier_init(self, seed: float = 0):
        generator = torch.Generator().manual_seed(seed)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p, generator=generator)
        return self
    
    def randn_noise(self, seed: float = 0):
        generator = torch.Generator().manual_seed(seed)
        for p in self.parameters():
            if len(p.shape) > 1:
                with torch.no_grad():
                    p += torch.randn(p.shape, generator=generator)
            elif len(p.shape) == 1:
                with torch.no_grad():
                    p += torch.randn(p.shape, generator=generator)
        return self
    
    def clamp(self, weights_bound, bias_bound):
        for mi, module in enumerate(self.layers):
            if hasattr(module, 'weight') and weights_bound:
                module.weight.data = module.weight.data.clamp(weights_bound[0][mi], weights_bound[1][mi])
            if hasattr(module, 'bias') and bias_bound:
                module.bias.data = module.bias.data.clamp(bias_bound[0][mi], bias_bound[1][mi])
        return self
    
    
    

class NN(nn.Module):
    def __init__(self, num_features, n_layers):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(num_features, n_layers[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(n_layers)-1):
            layers.append(nn.Linear(n_layers[i], n_layers[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(n_layers[-1],1))
        layers.append(nn.Sigmoid()) 
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
    
    def xavier_init(self, seed: float = 0):
        generator = torch.Generator().manual_seed(seed)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p, generator=generator)
        return self
    
    def clamp(self, weights_bound, bias_bound):
        for mi, module in enumerate(self.layers):
            if hasattr(module, 'weight') and weights_bound:
                module.weight.data = module.weight.data.clamp(weights_bound[0][mi], weights_bound[1][mi])
            if hasattr(module, 'bias') and bias_bound:
                module.bias.data = module.bias.data.clamp(bias_bound[0][mi], bias_bound[1][mi])
        return self