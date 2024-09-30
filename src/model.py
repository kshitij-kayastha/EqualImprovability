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
                module.bias.data = theta[-1]
        return self
    
    def get_theta(self):
        for module in self.layers:
            if hasattr(module, 'weight'):
                weights = module.weight.data
            if hasattr(module, 'bias'):
                bias = module.bias.data
        theta = torch.cat((weights[0], bias), 0)
        return theta
    
    
    def xavier_init(self):
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
        return self
    
    def bounded_init(self, weights_bound, bias_bound):
        for p in self.parameters():
            if len(p.shape) > 1:
                with torch.no_grad():
                    p += torch.randn(p.shape).clamp(weights_bound[0], weights_bound[1])
            elif len(p.shape) == 1:
                with torch.no_grad():
                    p += torch.randn(p.shape).clamp(bias_bound[0], bias_bound[1])
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