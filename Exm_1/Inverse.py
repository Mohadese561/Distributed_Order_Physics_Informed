"""
@author: Mohadese Ramezani
"""


# Import libraries
import torch
from torch import log, exp , pi , sin, cos
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torchaudio.functional import gain

import numpy as np
from scipy.special import gamma
from scipy import linalg

import colorama
from colorama import Fore, Style

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.set_default_dtype(torch.float)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    
def exact_u(X):
    x,t = X[:,[1]], X[:,[0]]
    return sin(2*pi*x) * (1 +  t ** 5)

def F(X):
    x,t = X[:,[1]], X[:,[0]]
    return torch.where(t != 1,((1/25) * pi**2*(1 + t**5)+(120*(-1 + t**beta)* t**(5 - beta))/log(t))*sin(2*pi*x) + 
                               2*pi * (1 + t**5) *cos(2*pi*x), 
                               4* pi *  cos(2*pi*x)+ (2/25) * (1500 * beta + pi**2) * sin(2*pi*x))

def given_func_w(alpha):
    return gamma(6-alpha)

# Generating training and test data sets
def data_train():
    t = torch.from_numpy(np.linspace(lb[0], ub[0], num_t)[:, None]).float()
    x_data = torch.from_numpy(np.linspace(lb[1], ub[1], num_x)[:, None]).float()
    return t, x_data

def data_test():
    t_test = np.linspace(lb[0], ub[0], t_test_N)[:, None]
    x_test = np.linspace(lb[1], ub[1], x_test_N)[:, None]
    t_star, x_star = np.meshgrid(t_test, x_test)
    test_data = np.hstack((t_star.flatten()[:, None], x_star.flatten()[:, None]))
    test_data = torch.from_numpy(test_data).float().to(device)
    test_exact = exact_u(test_data)

    return t_test, x_test, test_data, test_exact


#Neural Network
class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.lambda1 = nn.Parameter(torch.tensor(0.1))
        self.lambda2 = nn.Parameter(torch.tensor(0.0))
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        a = x
        for i in range(len(self.linear) - 1):
            a = self.activation(self.linear[i](a))
        return self.linear[-1](a)
    
#dPINN
class Model:
    def __init__(self, net, x_data, t, lb, ub,
                 test_data, test_exact, ha, beta,
                 ):
        self.net = net
        self.ha = ha
        self.beta = beta 
        
        self.x_data = x_data.to(device)
        self.t = t.to(device)
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        
        self.test_data = test_data.to(device)
        self.test_exact = test_exact.to(device)

        self.num_x = len(x_data)
        self.num_t = len(t)
        self.dt = (ub[0] - lb[0]) / (self.num_t - 1)
        
        self.coef = self.coef_d(np.arange(self.num_t))

        self.loss_records = {
            'initial': [], 'boundary': [], 'pde': [],
            'total': [], 'error': [], 'predictions': [],
            'lambda1': [], 'lambda2': [],
        }
        self.init_data()
        
        
    def coef_a(self, n, alpha):
        return self.dt ** (-alpha) * ((n + 1) ** (1 - alpha) - n ** (1 - alpha)) / gamma(2 - alpha) 


    def coef_d(self, n):
        Na = int(np.floor(self.beta / self.ha))
        alpha = (np.arange(1, Na + 1) - 0.5) * self.ha
        sum_ = 0
        for l in np.arange(Na):
            w = given_func_w(alpha[l])
            sum_ += w * self.coef_a(n, alpha[l])

        return self.ha * sum_

    def init_data(self):
        temp_t0 = torch.full((self.num_x, 1), self.t[0][0], device=device)
        temp_T = torch.full((self.num_x, 1), self.t[-1][0], device=device)
        self.tx_t0 = torch.cat((temp_t0, self.x_data), dim=1)
        self.tx_T = torch.cat((temp_T, self.x_data), dim=1)
        temp_t = self.t.clone().detach().to(dtype=torch.float32, device=device)
        x_data_repeated = self.x_data.repeat((self.num_t, 1))
        self.tx = torch.cat((temp_t.repeat_interleave(self.num_x).view(-1, 1), x_data_repeated), dim=1)

        temp_lb = torch.full((self.num_t, 1), self.lb[1], device=device)
        temp_ub = torch.full((self.num_t, 1), self.ub[1], device=device)
        self.tx_b1 = torch.cat((temp_t, temp_lb), dim=1)
        self.tx_b2 = torch.cat((temp_t, temp_ub), dim=1)

        self.u_x_b1 = exact_u(self.tx_b1)
        self.u_x_b2 = exact_u(self.tx_b2)
        self.u_t0 = exact_u(self.tx_t0)
        self.u_T = exact_u(self.tx_T)
        
        sigma = (noise * self.u_T.cpu().detach().numpy())**2
        mu = 0 
        s = np.random.normal(mu, sigma, self.u_T.shape)
        self.u_T += torch.from_numpy(s).cuda()
        
        self.F_tx = F(self.tx)
      

    def train_U(self, x):
        scaled_x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(scaled_x)
    
    
    def PDE_loss(self):
        coef = self.coef
        x = Variable(self.tx, requires_grad=True)
        u_pred = self.train_U(x)
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, [1]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [1]]

        u_n = u_pred.reshape(self.num_t, -1)
        Lu =  self.net.lambda1* u_xx.reshape(self.num_t, -1) - self.net.lambda2*u_x.reshape(self.num_t, -1) 
        F_n = self.F_tx.reshape(self.num_t, -1)

        loss = torch.tensor(0.0).to(device)

        for n in range(1, self.num_t):
            if n == 1:
                pre_Ui = (Lu[n] + F_n[n])/coef[0] + (coef[n-1] /coef[0]) * u_n[0]
            else:
                pre_Ui = ((Lu[n] + F_n[n])/coef[0] + (coef[n-1]/ coef[0]) * u_n[0]).to(device)
                for k in range(1, n):
                    pre_Ui += ((coef[n-k-1] - coef[n-k]) / coef[0]) * u_n[k]
            loss += torch.mean((pre_Ui - u_n[n]) ** 2)

        return loss

    def compute_loss(self):     
        loss_cond = torch.mean((self.train_U(self.tx_T) - self.u_T) ** 2)
        loss_initial = torch.mean((self.train_U(self.tx_t0) - self.u_t0) ** 2)
        self.loss_records['initial'].append([self.net.iter, loss_initial.item()])

        loss_boundary1 = torch.mean((self.train_U(self.tx_b1) - self.u_x_b1) ** 2)
        loss_boundary2 = torch.mean((self.train_U(self.tx_b2) - self.u_x_b2) ** 2)
        loss_boundary = loss_boundary1 + loss_boundary2
        self.loss_records['boundary'].append([self.net.iter, loss_boundary.item()])

        loss_pde = self.PDE_loss()
        self.loss_records['pde'].append([self.net.iter, loss_pde.item()])

        return loss_initial,  loss_boundary, loss_pde, loss_cond

    # computer backward loss
    def LBGFS_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss_initial, loss_boundary, loss_pde, loss_cond = self.compute_loss()
        total_loss = loss_initial + loss_boundary + loss_pde + loss_cond
        self.loss_records['total'].append([self.net.iter, total_loss.item()])
        total_loss.backward()
        self.loss_records['lambda1'].append([self.net.iter, self.net.lambda1.cpu().detach().numpy()])
        self.loss_records['lambda2'].append([self.net.iter, self.net.lambda2.cpu().detach().numpy()])   
        if self.net.iter % 100 == 0:
            print( 'Iteration: %.0f,  loss: %.5f , 𝜆_1 = %.4f, 𝜆_2 = %.4f' %
                    (
                        self.net.iter,
                        total_loss.item(),
                        self.net.lambda1,
                        self.net.lambda2,
                    )
                 )

        self.net.iter += 1
        predictions = self.train_U(test_data).cpu().detach().numpy()
        exact_values = self.test_exact.cpu().detach().numpy()
        error = np.linalg.norm(predictions - exact_values, 2) / np.linalg.norm(exact_values, 2)
        self.loss_records['error'].append([self.net.iter, error])

        if self.net.iter % 10 == 0:
            predictions = self.train_U(test_data).cpu().detach().numpy()
            self.loss_records['predictions'].append(predictions.tolist())

        return total_loss


    def train(self, LBGFS_epochs=50000):

        self.optimizer_LBGFS = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=LBGFS_epochs,
            max_eval=LBGFS_epochs,
            history_size= 50,
            tolerance_grad=1e-9,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        
        self.optimizer_LBGFS.step(self.LBGFS_loss)
        pred = self.train_U(test_data).cpu().detach().numpy()
        exact = self.test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print(Fore.BLUE + 'Test_L2error:' , '{0:.4e}'.format(error)+ Style.RESET_ALL)

        print('Training time: %.2f' % elapsed)

        return error, elapsed, self.LBGFS_loss().item()
    
if __name__ == '__main__':
    set_seed(1234)

    noise = 0.0

    n_layers = 5
    n_neurons = 20
    layers  = [2] + [n_neurons] * n_layers + [1]
    net = Net(layers).to(device)
    torch.nn.DataParallel(net)

    # Distributed-order parameters
    beta = 1/2
    ha = 1/100


    lb = np.array([0.0, 0.0]) # low boundary
    ub = np.array([1.0, 1.0]) # up boundary

    '''train data'''
    num_t = 80
    num_x = 50
    dt = ((ub[0] - lb[0]) / (num_t - 1))
    t, x_data = data_train()

    '''test data'''
    t_test_N = 100
    x_test_N = 100

    t_test, x_test, test_data, test_exact = data_test()


    '''Train''' 
    model = Model(
        net=net,
        x_data=x_data,
        t=t,
        lb=lb,
        ub=ub,
        test_data = test_data,
        test_exact=test_exact,
        ha = ha,
        beta = beta,
    )


    model.train(LBGFS_epochs=50000)
