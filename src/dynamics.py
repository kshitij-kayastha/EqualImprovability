from scipy.integrate import quad
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
import seaborn as sns
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.stats import wasserstein_distance
from functools import partial
from tqdm import tqdm 
from torch.utils.data import DataLoader


def drawTrajectory(dfs, y = 'disp', df_labels = ['ERM', 'DP', 'EI', 'BE', 'ER'], 
                   ylabelpad = 0, legend_loc = (0.58, 0.52), title = r'$A$-conscious',
                  colors = ['#073B4C','#FFD166','#06D6A0','#118AB2', '#DD3497', '#AE017E', '#7A0177', '#49006A'], 
                  markers = ['v', 'o', 'x', '1', 's', '>', '.'], zero_line = False, n_iters = 20, file_name=None):
    width = 5
    height = width

    plt.rc('font', family='serif', serif='times new roman')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    plt.rc('axes', labelsize=28)
    plt.rc('axes', linewidth=1)
    mpl.rcParams['patch.linewidth']=0.5 #width of the boundary of legend

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True) #plot two subgraphs in one figure, 1 by 2 size, sharing y-axis
    fig.subplots_adjust(left=.13, bottom=.25, right=0.97, top=0.88, wspace=0.05) #margin of the figure
    fig.set_size_inches(width, height) #exact size of the figure
    
    for i in range(len(dfs)):
        dfs[i][:n_iters].plot.line(y = y, ax = ax, label = df_labels[i], color = colors[i], marker = markers[i])
    ax.set_xlabel('')
    ax.set_title(title, fontsize = 25)
    ax.legend(fontsize = 16, bbox_to_anchor=legend_loc)
    
    if zero_line: 
        xmin, xmax = ax.get_xlim()
        ax.hlines(y = 0, color = 'gray', linestyle = '--', xmin = xmin, xmax = xmax)

    #a nice trick to add common x_label and y_label
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("Iteration")
    if y == 'disp':
        ylabel = 'Underlying DP Disp'
    elif y == 'wd':
        ylabel = "Distance"
    elif y == 'er':
        ylabel = "Error Rate"
    elif y == 'cdp':
        ylabel = 'Classification DP Disp'
    elif y == 'cef':
        ylabel = 'Classification Effort-encourging Unfairness'

    plt.ylabel(ylabel, labelpad = ylabelpad)
    # lines, labels = ax1.get_legend_handles_labels()
    # fig.legend(lines, labels, loc = 'upper center')
    if file_name: 
        plt.savefig('figures/%s.pdf' % file_name)
        print('File saved in figures/%s.pdf!' % file_name)




class populationDynamics_gaussian(object):
    def __init__(self, mean0=0, std0=1, mean1=1, std1=0.9, frac=None, eps=.6, lazy_pos = True, group0_frac = 0.5):
        self.init_data = [{'mean': mean0, 'std': std0}, 
                         {'mean': mean1, 'std': std1}]
        self.frac, self.eps = frac, eps
        self.lazy_pos = lazy_pos
        self.err_thres = frac / 2
        self.group_frac = np.array([group0_frac, 1-group0_frac])
        
    def dataPrint(self, data):
        mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
        return (r'$\mu^{(0)}=$%.1f, $\sigma^{(0)}=$%.1f' % (mu0, sigma0), r'$\mu^{(1)}=$%.1f, $\sigma^{(1)}=$%.1f' % (mu1, sigma1))

    def plot(self, data, truebs, bs = None, title = None):
        fig, ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, sharex = True)
        labels = self.dataPrint(data)
        for a in range(2):
            mu, sigma = data[a]['mean'], data[a]['std']
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            ax[a].plot(x, norm.pdf(x, mu, sigma))
            ax[a].set_xlabel(labels[a], fontsize = 18)
            ymin, ymax = ax[a].get_ylim()
            ax[a].vlines(x = truebs[a], color = 'g', linestyle='-.', ymin = ymin, ymax = ymax)
            if not bs is None: ax[a].vlines(x = bs[a], color = 'm', linestyle = '-', ymin = ymin, ymax = ymax)
        plt.title(title)
    
    
    # Calculate eps(x) in EI paper page 7, last line
    def effort(self, x, boundary, effort_pos = 0, eps = None):
        # eps below is the beta in the EI paper
        if eps is None: 
            eps = self.eps 
        if x >= boundary:
            return effort_pos
        else:
            return 1/(boundary - x + eps)**2
    
    def WassersteinDistance(self, data):
        # https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
        mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
        return (mu0 - mu1)**2 + (sigma0 - sigma1)**2
    
    # Distance
    # quad means integral from a to b: quad(function, a, b)
    def distance(self, data): 
        mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
        f = lambda x: abs(norm.pdf(x, mu0, sigma0) - norm.pdf(x, mu1, sigma1))
        return quad(f, a=-np.inf, b=np.inf)[0]/2

    # Calculate Error Rate
    def errorRate(self, data, truebs, bs):
        mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
        # true boundary
        tb = truebs[0]
        e0 = abs(norm.cdf(tb, mu0, sigma0) - norm.cdf(bs[0], mu0, sigma0))
        e1 = abs(norm.cdf(tb, mu1, sigma1) - norm.cdf(bs[1], mu1, sigma1))
        return e0*self.group_frac[0]+e1*self.group_frac[1]
    
    # True Boundary
    def trueBoundary(self, data, thres = 1e-3):
        if self.frac:
            mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
            # lower bound and upper bound for binary search
            l, u = min(mu0-3*sigma0, mu1-3*sigma1), max(mu0+3*sigma0, mu1+3*sigma1)
            while True:
                p = (l+u)/2
                frac = 1-(norm.cdf(p, mu0, sigma0) + norm.cdf(p, mu1, sigma1))/2
                if abs(frac - self.frac) < thres or (u-l) < 1e-10: break
                if frac > self.frac:
                    l = p
                else:
                    u = p
            return np.ones(2) * p
        else:
            return np.zeros(2)
        
    # Demographic Parity
    def dpDisp(self, data, bs, is_abs = True):
        pos = []
        for z in range(2):
            mu, sigma = data[z]['mean'], data[z]['std']
            pos.append(1-norm.cdf(bs[z], mu, sigma))
        return abs(pos[1] - pos[0]) if is_abs else pos[1] - pos[0]
        
    # EI Disparity
    def eiDisp(self, data, bs, delta, is_abs = True):
        pos = []
        for z in range(2):
            mu, sigma = data[z]['mean'], data[z]['std']
            
            # CDF for normal distribution with mean mu and std dev sigma
            # (CDF(x+d) - CDF(x)) / CDF(x)
            pos.append((norm.cdf(bs[z], mu + delta, sigma) - norm.cdf(bs[z], mu, sigma)) / norm.cdf(bs[z], mu, sigma))
            
        return abs(pos[1] - pos[0]) if is_abs else pos[1] - pos[0]

    # Bounded Effort
    def beDisp(self, data, bs, delta, is_abs = True):
        pos = []
        for z in range(2):
            mu, sigma = data[z]['mean'], data[z]['std']
            pos.append(norm.cdf(bs[z], mu + delta, sigma)-norm.cdf(bs[z], mu, sigma))
        return abs(pos[1] - pos[0]) if is_abs else pos[1] - pos[0]

    # Equal Recourse
    def erDisp(self, data, bs, delta, is_abs = True):
        efforts = []
        for z in range(2):
            mu, sigma = data[z]['mean'], data[z]['std']
            effort = quad(lambda x: (bs[z]-x) * norm.pdf(x, mu, sigma), -np.inf, bs[z])[0]
            efforts.append(effort / norm.cdf(bs[z], mu, sigma))
        return abs(efforts[0] - efforts[1]) if is_abs else efforts[1] - efforts[0]

    # ILFCR
    def ilerDisp(self, data, bs, delta, is_abs = True):
        efforts1, efforts2 = [], []
        u0 = (bs[0] - data[0]['mean'])/data[0]['std']
        u1 = (bs[1] - data[1]['mean'])/data[1]['std']
        u = min(u0, u1)
        for a in range(2):
            mu, sigma = data[a]['mean'], data[a]['std']
            effort1 = bs[a] - (data[a]['mean'] - 3*data[a]['std'])/data[a]['std']
            effort2 = bs[a] - (data[a]['mean'] - u*data[a]['std'])/data[a]['std']
            efforts1.append(effort1 / norm.cdf(bs[a], mu, sigma))
            efforts2.append(effort2 / norm.cdf(bs[a], mu, sigma))
        return max(abs(efforts1[0] - efforts1[1]), abs(efforts2[0] - efforts2[1]))

    
    # Boundary found by Demographic Parity
    def DPBoundary(self, data, truebs, c = 0, thres = 1e-3):
        tb = truebs[0]
        mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']

        func = lambda x: self.dpDisp(data, x)
        # cons = ({'type': 'ineq', 'fun': lambda x: c-self.dpDisp(data, x)})
        cons = ({'type': 'ineq', 'fun': lambda x: self.err_thres-self.errorRate(data, truebs, x)})
        bnds = [(mu0-3*sigma0, mu0+3*sigma0), (mu1-3*sigma1, mu1+3*sigma1)]
        res = minimize(func, truebs, method = 'SLSQP', bounds = bnds, constraints = cons)
        return res.x

    # Boundary found by EI
    def EIBoundary(self, data, truebs, delta, c = 0, thres = 1e-3):
        tb = truebs[0]
        mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
        
        func = lambda x: self.eiDisp(data, x, delta)
        cons = ({'type': 'ineq', 'fun': lambda x: self.err_thres-self.errorRate(data, truebs, x)})
        bnds = [(mu0-3*sigma0, mu0+3*sigma0), (mu1-3*sigma1, mu1+3*sigma1)]
        res = minimize(func, truebs, method = 'SLSQP', bounds = bnds, constraints = cons)
        return res.x

    # Boundary found by Bounded Effort
    def BEBoundary(self, data, truebs, delta, c = 0, thres = 1e-3):
        tb = truebs[0]
        mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
        
        func = lambda x: self.beDisp(data, x, delta)
        cons = ({'type': 'ineq', 'fun': lambda x: self.err_thres-self.errorRate(data, truebs, x)})
        bnds = [(mu0-3*sigma0, mu0+3*sigma0), (mu1-3*sigma1, mu1+3*sigma1)]
        res = minimize(func, truebs, method = 'SLSQP', bounds = bnds, constraints = cons)
        return res.x

    # Boundary found by Equal Recourse
    def ERBoundary(self, data, truebs, delta, c = 0, thres = 1e-3):
        tb = truebs[0]
        mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
        
        func = lambda x: self.erDisp(data, x, delta)
        cons = ({'type': 'ineq', 'fun': lambda x: self.err_thres-self.errorRate(data, truebs, x)})
        bnds = [(mu0-3*sigma0, mu0+3*sigma0), (mu1-3*sigma1, mu1+3*sigma1)]
        res = minimize(func, truebs, method = 'SLSQP', bounds = bnds, constraints = cons)
        return res.x

    def ILERBoundary(self, data, truebs, delta, c = 0, thres = 1e-3):
        tb = truebs[0]
        mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
        
        func = lambda x: self.ilerDisp(data, x, delta)
        cons = ({'type': 'ineq', 'fun': lambda x: self.err_thres-self.errorRate(data, truebs, x)})
        bnds = [(mu0-3*sigma0, mu0+3*sigma0), (mu1-3*sigma1, mu1+3*sigma1)]
        res = minimize(func, truebs, method = 'SLSQP', bounds = bnds, constraints = cons)
        return res.x

    # update mean and variance of data
    def update(self, data, bs, lazy_pos = True):
        mean_efforts, updated_var = np.zeros(2), np.zeros(2)
        
        for z in range(2):
            if lazy_pos:
                mean_efforts[z] = quad(lambda x: norm.pdf(x, data[z]['mean'], data[z]['std']) \
                                * self.effort(x, bs[z], eps = 1/data[z]['std'] ** .5), -np.inf, np.inf)[0] 
                updated_var[z] = quad(lambda x: norm.pdf(x, data[z]['mean'], data[z]['std']) \
                                 * (x+self.effort(x, bs[z], eps = 1/data[z]['std'] ** .5) - data[z]['mean'] - mean_efforts[z]) ** 2, \
                                  -np.inf, np.inf)[0]
            else:
                effort_neg = quad(lambda x: norm.pdf(x, data[z]['mean'], data[z]['std']) \
                                    * self.effort(x, bs[z], eps = 1/data[z]['std'] ** .5), -np.inf, bs[z])[0] \
                                    /norm.cdf(bs[z], data[z]['mean'], data[z]['std'])
                effort_pos = effort_neg
                mean_efforts[z] = effort_neg
                updated_var[z] = quad(lambda x: norm.pdf(x, data[z]['mean'], data[z]['std']) \
                                    * (x+self.effort(x, bs[z], effort_pos, eps = 1/data[z]['std'] ** .5) - data[z]['mean'] - mean_efforts[z]) ** 2, \
                                    -np.inf, np.inf)[0]
            data[z] = {'mean': data[z]['mean'] + mean_efforts[z], 'std': updated_var[z]**.5}
        return data

    # Find $\delta_{t}$
    # data: x_t
    # bs: boundary (threshold)
    def selectDelta(self, data, bs, mode = 'median'):
        if mode == 'median':
            mu0, sigma0, mu1, sigma1 = data[0]['mean'], data[0]['std'], data[1]['mean'], data[1]['std']
            neg_frac = norm.cdf(bs[0], mu0, sigma0) + norm.cdf(bs[1], mu1, sigma1)
            func = lambda x: norm.cdf(x, mu0-bs[0], sigma0) + norm.cdf(x, mu1-bs[1], sigma1) / neg_frac - 0.5
            cons = ({'type': 'eq', 'fun': func})
            bnds = [(None, 0)]
            res = minimize(func, 0, method = 'SLSQP', bounds = bnds, constraints = cons)
            return self.effort(res.x, 0)
            
        elif mode == 'mean':
            mean_efforts, neg_frac = np.zeros(2), np.zeros(2)
            for a in range(2):
                mean_efforts[a] = quad(lambda x: norm.pdf(x, data[a]['mean'], data[a]['std']) \
                                    * self.effort(x, bs[a], eps = 1/data[a]['std'] ** .5), -np.inf, bs[a])[0] 
                neg_frac[a] = norm.cdf(bs[a], data[a]['mean'], data[a]['std'])
            # print((mean_efforts[0] * neg_frac[0] + mean_efforts[1] * neg_frac[1]) / (neg_frac[0] + neg_frac[1]))
            return (mean_efforts[0] * neg_frac[0] + mean_efforts[1] * neg_frac[1]) / (neg_frac[0] + neg_frac[1])
        else:
            raise ValueError("Unexpected delta select mode %s! Supported mode: \"mean\" and \"median\"." % mode)
        

    def run(self, mode = 'true', n_iter = 20, delta = 0.5, c = 0, thres = .001, select_delta = False, plot = True):
        # ERM
        if mode == 'true':
            data = copy.deepcopy(self.init_data)
            disp, wd, cdp, cef = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                disp[i], wd[i], cdp[i], cef[i] = self.dpDisp(data, truebs), self.distance(data), self.dpDisp(data, truebs), self.eiDisp(data, truebs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EI Disp: %.3f, Wasserstein Distance: %.3f' % (
                        i, disp[i], cdp[i], cef[i], wd[i])
                if plot: self.plot(data, truebs, title = text)
                data = self.update(data, truebs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'cdp': cdp, 'cef': cef})
                
        # Demographic Parity
        elif mode == 'dp':
            data = copy.deepcopy(self.init_data)
            disp, wd, er, cdp, cef = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                bs = self.DPBoundary(data, truebs, c, thres)
                disp[i], wd[i], er[i], cdp[i], cef[i] = self.dpDisp(data, truebs), self.distance(data), self.errorRate(data, truebs, bs), self.dpDisp(data, bs), self.eiDisp(data, bs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EI Disp: %.3f, Wasserstein Distance: %.3f, Error Rate: %.3f' % (
                        i, disp[i], cdp[i], cef[i] , wd[i], er[i])
                if plot: self.plot(data, truebs, bs, title = text)
                data = self.update(data, bs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'er':er, 'cdp':cdp, 'cef':cef})

        # EI (INTERESTED IN THIS)
        elif mode == 'ei':
            data = copy.deepcopy(self.init_data)
            disp, wd, er, cdp, cef = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                if select_delta: 
                    delta = self.selectDelta(data, truebs, select_delta)
                
                bs = self.EIBoundary(data, truebs, delta, c, thres)
                
                disp[i], wd[i], er[i], cdp[i], cef[i] = self.dpDisp(data, truebs), self.distance(data), self.errorRate(data, truebs, bs), self.dpDisp(data, bs), self.eiDisp(data, bs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EI Disp: %.3f, Wasserstein Distance: %.3f, Error Rate: %.3f' % (
                        i, disp[i], cdp[i], cef[i] , wd[i], er[i])
                if plot: 
                    self.plot(data, truebs, bs, title = text)
                
                data = self.update(data, bs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'er':er, 'cdp':cdp, 'cef':cef})

        # Bounded Effort
        elif mode == 'be':
            data = copy.deepcopy(self.init_data)
            disp, wd, er, cdp, cef, cbe = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                if select_delta: delta = self.selectDelta(data, truebs, select_delta)
                bs = self.BEBoundary(data, truebs, delta, c, thres)
                disp[i], wd[i], er[i], cdp[i], cef[i], cbe[i] = self.dpDisp(data, truebs), self.distance(data), self.errorRate(data, truebs, bs), self.dpDisp(data, bs), self.eiDisp(data, bs, delta), self.beDisp(data, bs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EI Disp: %.3f, Classification BE Disp: %.3f, Wasserstein Distance: %.3f, Error Rate: %.3f' % (
                        i, disp[i], cdp[i], cef[i], cbe[i], wd[i], er[i])
                if plot: self.plot(data, truebs, bs, title = text)
                data = self.update(data, bs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'er':er, 'cdp':cdp, 'cef':cef, 'cbe': cbe})

        # Equal Recourse
        elif mode == 'er':
            data = copy.deepcopy(self.init_data)
            disp, wd, er, cdp, cef, cer = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                if select_delta: delta = self.selectDelta(data, truebs, select_delta)
                bs = self.ERBoundary(data, truebs, delta, c, thres)
                disp[i], wd[i], er[i], cdp[i], cef[i], cer[i] = self.dpDisp(data, truebs), self.distance(data), self.errorRate(data, truebs, bs), self.dpDisp(data, bs), self.eiDisp(data, bs, delta), self.erDisp(data, bs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EI Disp: %.3f, Classification ER Disp: %.3f, Wasserstein Distance: %.3f, Error Rate: %.3f' % (
                        i, disp[i], cdp[i], cef[i], cer[i], wd[i], er[i])
                if plot: self.plot(data, truebs, bs, title = text)
                data = self.update(data, bs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'er':er, 'cdp':cdp, 'cef':cef, 'cer': cer})

        # ILFCR
        elif mode == 'iler':
            data = copy.deepcopy(self.init_data)
            disp, wd, er, cdp, cef, cer = np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1), np.zeros(n_iter + 1)
            for i in range(n_iter+1):
                truebs = self.trueBoundary(data, thres)
                if select_delta: delta = self.selectDelta(data, truebs, select_delta)
                bs = self.ILERBoundary(data, truebs, delta, c, thres)
                disp[i], wd[i], er[i], cdp[i], cef[i], cer[i] = self.dpDisp(data, truebs), self.distance(data), self.errorRate(data, truebs, bs), self.dpDisp(data, bs), self.eiDisp(data, bs, delta), self.ilerDisp(data, bs, delta)
                text = 'Iteration %d: Underlying DP Disp: %.3f, Classification DP Disp: %.3f, Classification EI Disp: %.3f, Classification ER Disp: %.3f, Wasserstein Distance: %.3f, Error Rate: %.3f' % (
                        i, disp[i], cdp[i], cef[i], cer[i], wd[i], er[i])
                if plot: self.plot(data, truebs, bs, title = text)
                data = self.update(data, bs)
            return pd.DataFrame({'disp':disp, 'wd':wd, 'er':er, 'cdp':cdp, 'cef':cef, 'ciler': cer})

        else:
            raise ValueError("Unexpected mode %s! Supported mode: \"true\", \"dp\", and \"ef\"." % mode)