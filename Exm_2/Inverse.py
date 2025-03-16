"""
@authors: Mohadese Ramezani & Maryam Mohammadi
"""


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

from matplotlib.path import Path
from pyDOE import lhs


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
    t, x, y = X[:, [0]], X[:, [1]], X[:, [2]]
    return (cos(x) + cos(y)) * (1 + t**(1/2))

def F(X):
    t, x, y = X[:, [0]], X[:, [1]], X[:, [2]]
    return torch.where(t != 1, (2 * (1 + t**(1/2)) + pi**(1/2) * (-1 + t) /(2* t**(1/2) * log(t)) ) * (cos(x) + cos(y))  - (1 + t**(1/2)) * (sin(x) + sin(y)),
                      ((pi**(1/2)+ 8)* (cos(x) + cos(y)) - 4 * (sin(x) + sin(y))) /2)


def given_func_w(alpha):
    return gamma(3/2-alpha)


# create geometry and generating trian and test data sets
def define_butterfly_boundary(a=4, b=3, num_points=1000):
    theta = np.linspace(0, 2 * np.pi, num_points)
    R = np.exp(np.cos(theta)) - np.cos(a * theta) + b * (np.sin(theta / 2) ** 5)
    x, y = R * np.cos(theta), R * np.sin(theta)
    scaling_factor = 3  # because the current range is from -3 to 3
    return x / scaling_factor, y / scaling_factor

def generate_boundary_points(n_points, a=4, b=3, lhs_sampling=True):
    theta = lhs(1, samples=n_points).flatten() * 2 * np.pi if lhs_sampling else np.linspace(0, 2 * np.pi, n_points)
    R = np.exp(np.cos(theta)) - np.cos(a * theta) + b * (np.sin(theta / 2) ** 5)
    R = R/3
    return R * np.cos(theta), R * np.sin(theta)

def generate_interior_points(x, y, n_points, shrink_factor=0.97, lhs_sampling=True):
    center_x, center_y = np.mean(x), np.mean(y)
    shrunken_x = center_x + shrink_factor * (x - center_x)
    shrunken_y = center_y + shrink_factor * (y - center_y)
    shrunken_path = Path(np.vstack((shrunken_x, shrunken_y)).T)

    if lhs_sampling:
        lhs_points = lhs(2, samples=n_points * 10)
        lhs_points[:, 0] = lhs_points[:, 0] * (np.max(x) - np.min(x)) + np.min(x)
        lhs_points[:, 1] = lhs_points[:, 1] * (np.max(y) - np.min(y)) + np.min(y)
        interior_points = lhs_points[shrunken_path.contains_points(lhs_points)]
        return interior_points[:n_points]   

    x_grid, y_grid = np.linspace(np.min(x), np.max(x), n_points), np.linspace(np.min(y), np.max(y), n_points)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    interior_points = grid_points[shrunken_path.contains_points(grid_points)]
    
    return interior_points


def data_train(x, y):
    t = torch.linspace(0, T, t_N).view(-1, 1).float()
    x_boundary, y_boundary = generate_boundary_points(num_boundary_points)
    boundary_points = np.column_stack((x_boundary, y_boundary))
    points_inside = generate_interior_points(x, y, num_interior_points)
    
    return t, points_inside, boundary_points

def data_test(x, y):
    t = torch.linspace(0, T, test_N).view(-1, 1).float()
    x_boundary, y_boundary = generate_boundary_points(num_boundary_points, lhs_sampling=False)
    boundary_points = np.column_stack((x_boundary, y_boundary))
    points_inside = generate_interior_points(x, y, num_test_points, lhs_sampling=False)
    xy = torch.from_numpy(np.vstack((points_inside, boundary_points)))
    xy_test = xy.repeat(len(t), 1)
    t_vector_test = t.repeat_interleave(len(xy)).view(-1, 1)
    txy_test = torch.cat((t_vector_test, xy_test), dim=1).float().to(device)

    with torch.no_grad():
        txy_test_exact = exact_u(txy_test)

    return xy, txy_test, txy_test_exact


#Neural Network
class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.lambda1 = nn.Parameter(torch.tensor(0.0))
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
    def __init__(self, net, xy, xy_bounds, t, txy_test, txy_test_exact, ha, beta):

        self.net = net
        self.ha = ha
        self.beta = beta
        

        self.xy_bounds = torch.from_numpy(xy_bounds).to(device, dtype=torch.float32)
        self.x_N = len(xy_bounds)
        self.xy = torch.from_numpy(xy).to(device, dtype=torch.float32)
        self.xy_N = len(xy)

        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        
        self.t = t.to(device, dtype=torch.float32)
        self.t_N = len(t)
        self.dt = ((ub[0] - lb[0]) / (self.t_N - 1))

        self.txy_test = txy_test.to(device, dtype=torch.float32)
        self.txy_test_exact = txy_test_exact.to(device, dtype=torch.float32)

        self.coef = self.coef_d(np.arange(self.t_N))
        self.loss_records = {key: [] for key in ['initial', 'boundary', 'pde', 'total', 'error', 
                                                 'predictions','lambda1', 'lambda2']}
        self.init_data()

    def coef_a(self, n, alpha):
        return self.dt ** (-alpha) * ((n + 1) ** (1 - alpha) - n ** (1 - alpha)) / gamma(2 - alpha)

    def coef_d(self, n):
        Na = int(np.floor(self.beta / self.ha))
        alpha = (np.arange(1, Na + 1) - 0.5) * self.ha
        sum_ = 0
        for l in range(Na):
            w = given_func_w(alpha[l])
            sum_ += w * self.coef_a(n, alpha[l])
        return self.ha * sum_

    def init_data(self):
        def create_full_txy(t, xy, N):
            t_repeated = torch.repeat_interleave(t[:, 0], N).view(-1, 1).to(device)
            xy_repeated = xy.repeat(len(t), 1)
            return torch.cat((t_repeated, xy_repeated), dim=1)

        self.xyb = torch.cat((self.xy, self.xy_bounds), dim=0)
        self.txy_t0 = torch.cat((torch.full_like(torch.zeros(self.xy_N + self.x_N, 1), self.t[0][0], device=device), self.xyb), dim=1)
        self.txy_T = torch.cat((torch.full_like(torch.zeros(self.xy_N + self.x_N, 1), self.t[-1][0], device=device), self.xyb), dim=1)
        self.txy = create_full_txy(self.t, self.xyb, self.xy_N + self.x_N)

        # Boundary conditions
        self.txy_bounds = create_full_txy(self.t, self.xy_bounds, self.x_N)
        
        
        self.u_bounds = exact_u(self.txy_bounds)       
        self.u_t0 = exact_u(self.txy_t0)
        
        if noise == 0.0:
            self.u_T = exact_u(self.txy_T)
        else:
            self.u_T = exact_u(self.txy_T)
            sigma = (noise * self.u_T.cpu().detach().numpy())**2
            mu = 0 
            s = np.random.normal(mu, sigma, self.u_T.shape)
            self.u_T += torch.from_numpy(s).cuda()
        
        self.F_txy = F(self.txy)
        

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H)

    def PDE_loss(self):
        coef = self.coef
        x = Variable(self.txy, requires_grad=True)

        u_pred = self.train_U(x).to(device)
        u_x_y = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_x, u_y = u_x_y[:,[1]], u_x_y[:,[2]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [1]]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [2]]

        u_n = u_pred.reshape(self.t_N, -1)
        Lu = self.net.lambda1*(u_xx.reshape(self.t_N, -1) + u_yy.reshape(self.t_N, -1)) - self.net.lambda2*(u_x.reshape(self.t_N, -1) + u_y.reshape(self.t_N, -1))
        F_n = self.F_txy.reshape(self.t_N, -1)

        loss = torch.tensor(0.0, device=device)

        for n in range(1, self.t_N):
            pre_Ui = ((Lu[n] + F_n[n]) / coef[0] + (coef[n-1] / coef[0]) * u_n[0]).to(device)
            if n > 1:
                for k in range(1, n):
                    pre_Ui += ((coef[n-k-1] - coef[n-k]) / coef[0]) * u_n[k]
            loss += torch.mean((pre_Ui - u_n[n]) ** 2)

        return loss

    def compute_loss(self):
        loss_cond = torch.mean((self.train_U(self.txy_T) - self.u_T) ** 2)
        
        u_pred_t0 = self.train_U(self.txy_t0)
        loss_initial = torch.mean((u_pred_t0 - self.u_t0) ** 2)
        self.loss_records['initial'].append([self.net.iter, loss_initial.item()])

        # Boundary condition losses
        u_pred = self.train_U(self.txy_bounds)
        loss_boundary = torch.mean((u_pred - self.u_bounds) ** 2)
        self.loss_records['boundary'].append([self.net.iter, loss_boundary.item()])

        # PDE loss
        loss_pde = self.PDE_loss()
        self.loss_records['pde'].append([self.net.iter, loss_pde.item()])

        return loss_initial, loss_boundary, loss_pde, loss_cond


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
        
        with torch.no_grad():
            predictions = self.train_U(self.txy_test).cpu().numpy()
            exact_values = self.txy_test_exact.cpu().numpy()
            error = np.linalg.norm(predictions - exact_values, 2) / np.linalg.norm(exact_values, 2)
            self.loss_records['error'].append([self.net.iter, error])

            if self.net.iter % 10 == 0:
                predictions = self.train_U(self.txy_test).cpu().numpy()
                self.loss_records['predictions'].append(predictions.tolist())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return total_loss

    def train_lbfgs(self, LBGFS_epochs):
        self.max_iterations = LBGFS_epochs
        self.optimizer_LBGFS = optim.LBFGS(
            self.net.parameters(),
            lr=1,
            max_iter=LBGFS_epochs,
            max_eval=LBGFS_epochs,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        self.optimizer_LBGFS.step(self.LBGFS_loss)
        pred = self.train_U(self.txy_test).cpu().numpy()
        exact = self.txy_test_exact.cpu().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        print(Fore.BLUE + 'Test_L2error:', '{0:.4e}'.format(error) + Style.RESET_ALL)

        return error
    
    def train(self, LBGFS_epochs):
        error = self.train_lbfgs(LBGFS_epochs)
        return error

    
if __name__ == '__main__':
    set_seed(1234)

    noise = 0.0

    n_layers = 5
    n_neurons = 60
    layers  = [3] + [n_neurons] * n_layers + [1]
    net = Net(layers).to(device)
    torch.nn.DataParallel(net)

    # Distributed-order parameters
    beta = 1
    ha = 1/100
    T = 1

    lb = np.array([0.0, -1.0, -1.0]) # low boundary
    ub = np.array([1.0, 1.0, 1.0]) # up boundary
    
    
    '''train data'''
    t_N = 10
    num_boundary_points = 50
    num_interior_points = 100    

    x, y = define_butterfly_boundary()
    t, xy, xy_bounds = data_train(x,y)

    '''test data'''
    test_N = 100
    num_boundary_points = 100
    num_test_points = 20
    txy_test, txy_test_exact = data_test(x,y)


    '''Train'''
    model = Model(
            net=net,
            xy=xy,
            xy_bounds=xy_bounds,
            t=t,
            txy_test=txy_test,
            txy_test_exact=txy_test_exact,
            ha=ha,
            beta=beta
        )


    model.train(LBGFS_epochs=20000)
