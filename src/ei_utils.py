import numpy as np

def pareto_frontier(A, B):
    temp = np.column_stack((A, B))
    is_dominated = np.ones(temp.shape[0], dtype=bool)
    
    for i, c in enumerate(temp):
        if is_dominated[i]:
            is_dominated[is_dominated] = np.any(temp[is_dominated] < c, axis=1)
            is_dominated[i] = True
    return is_dominated    


def ei_disparity(n_eyz, each_z = False):
    '''
    Equal improvability disparity: max_z{|P(yhat_max=1|z=z,y_hat=0)-P(yhat_max=1|y_hat=0)|}

    Parameters
    ----------
    n_eyz : dictionary
        #(yhat_max=e,yhat=y,z=z)
    each_z : Bool
        Returns for each sensitive group or max
    '''
    z_set = list(set([z for _,_, z in n_eyz.keys()]))
    
    ei = []
    if sum([n_eyz[(1,0,z)]+n_eyz[(0,0,z)] for z in z_set])==0:
        p10 = 0
    else:
        p10 = sum([n_eyz[(1,0,z)] for z in z_set]) / sum([n_eyz[(1,0,z)]+n_eyz[(0,0,z)] for z in z_set])
    for z in z_set:
        if n_eyz[(1,0,z)] == 0: 
            ei_z = 0
        else:
            ei_z = n_eyz[(1,0,z)] / (n_eyz[(0,0,z)] + n_eyz[(1,0,z)])
        ei.append(abs(ei_z-p10))
    if each_z:
        return ei
    else:
        return max(ei)

def model_performance(Y, Z, Y_hat, Y_hat_max, tau):
    Y_pred = (Y_hat>=tau)*1
    Y_pred_max = (Y_hat_max>=tau)*1
    acc = np.mean(Y==Y_pred)
    
    n_eyz = {}
    for y_max in [0,1]: 
        for y in [0,1]:
            for z in [0,1]:
                n_eyz[(y_max,y,z)] = np.sum((Y_pred_max==y_max)*(Y_pred==y)*(Z==z))
    return acc, ei_disparity(n_eyz)


def generate_grid(center, width, n=100, ord=None):
    axes = [np.linspace(center[i]-width, center[i]+width, n) for i in range(len(center))]
    grids = np.meshgrid(*axes)
    points = np.stack([grid.reshape(-1) for grid in grids]).T
    if ord:
        mask = np.linalg.norm(points-center, ord=ord, axis=1) <= (width+1e-5)
        points = points[mask]
    return np.unique(points, axis=0)