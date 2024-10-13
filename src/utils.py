import pickle
import numpy as np

def pareto_frontier(A, B):
    temp = np.column_stack((A, B))
    is_dominated = np.ones(temp.shape[0], dtype=bool)
    
    for i, c in enumerate(temp):
        if is_dominated[i]:
            is_dominated[is_dominated] = np.any(temp[is_dominated] < c, axis=1)
            is_dominated[i] = True
    return is_dominated

# def ei_disparity(n_eyz):
#     z_set = [0,1]
#     for z in z_set:
#         if (n_eyz[(1,z)] == 0): 
#             if z==0:
#                 ei_0 = 0
#             else:
#                 ei_1 = 0
#         else:
#             if z==0:
#                 ei_0 = n_eyz[(1,0)] / (n_eyz[(0,0)] + n_eyz[(1,0)])
#             else:
#                 ei_1 = n_eyz[(1,1)] / (n_eyz[(0,1)] + n_eyz[(1,1)])
#     return abs(ei_0-ei_1)


def ei_disparity(n_eyz):
    z_set = [0,1]
    ei = []
    
    if sum([n_eyz[(1,z)]+n_eyz[(0,z)] for z in z_set])==0:
        p10 = 0
    else:
        p10 = sum([n_eyz[(1,z)] for z in z_set]) / sum([n_eyz[(1,z)]+n_eyz[(0,z)] for z in z_set])
    
    for z in z_set:
        if (n_eyz[(1,z)] == 0): 
            ei_z = 0
        else:
            ei_z = n_eyz[(1,z)] / (n_eyz[(0,z)] + n_eyz[(1,z)])
        ei.append(abs(ei_z-p10))
    return max(ei)


def model_performance(Y, Z, Y_hat, Y_hat_max, tau):
    Y_pred = (Y_hat>=tau)*1
    Y_pred_max = (Y_hat_max>=tau)*1
    acc = np.mean(Y==Y_pred)
    
    n_eyz = {}
    for y_pred_max in [0,1]: 
        for z in [0,1]:
            n_eyz[(y_pred_max, z)] = np.sum((Y_pred_max==y_pred_max)*(Z==z))
    return acc, ei_disparity(n_eyz)


# save pickle 
def pdump(filepath, obj):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
       
# load pickle 
def pload(filepath):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj