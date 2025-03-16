"""
@authors: Mohadese Ramezani & Maryam Mohammadi
"""


import torch
from torch import log, exp , pi , sin, cos, sinh, cosh
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torchaudio.functional import gain

import numpy as np
from scipy.special import gamma
from scipy import linalg

from random import uniform
from functools import partial

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
    t, x, y, z = X[:, [0]], X[:, [1]], X[:, [2]], X[:,[3]]
    return (1- (x**2 + y**2 + z**2)) * (1+t**3)

def F(X):
    t, x, y, z = X[:, [0]], X[:, [1]], X[:, [2]], X[:,[3]]
    return torch.where(t != 1, -2 * (1+t**3) * (-3 + x + y + z) - (6* (-1 + t) * t**2 * (-1 + x**2 + y**2 + z**2))/log(t),
                            -2* (-9 + 2 *x + 3 *x**2 + 2 *y + 3* y**2 + 2* z + 3 *z**2))

def given_func_w(alpha):
    return gamma(4-alpha)


#Create Geometry and generating train and test data sets
def define_sphere_boundary(radius=0.5, num_points=1000):
    phi = np.linspace(0, np.pi, num_points)  
    theta = np.linspace(0, 2 * np.pi, num_points) 
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return x.ravel(), y.ravel(), z.ravel()

def generate_sphere_boundary_points(n_points, radius=0.5, lhs_sampling=True):
    if lhs_sampling:
        samples = lhs(2, samples=n_points)
        phi = samples[:, 0] * np.pi  
        theta = samples[:, 1] * 2 * np.pi  
    else:
        phi = np.linspace(0, np.pi, n_points)
        theta = np.linspace(0, 2 * np.pi, n_points)
        phi, theta = np.meshgrid(phi, theta)
        phi, theta = phi.ravel(), theta.ravel()

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return x, y, z

def generate_interior_points_sphere(radius=0.5, n_points=10, lhs_sampling=True):
    if lhs_sampling:
        samples = lhs(3, samples=n_points)
        r = radius * samples[:, 0] ** (1/3)  
        phi = samples[:, 1] * np.pi  
        theta = samples[:, 2] * 2 * np.pi  
    else:
        r = np.linspace(0, radius, n_points)
        phi = np.linspace(0, np.pi, n_points)
        theta = np.linspace(0, 2 * np.pi, n_points)
        r, phi, theta = np.meshgrid(r, phi, theta)
        r, phi, theta = r.ravel(), phi.ravel(), theta.ravel()

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    # Stack the coordinates into a 2D array
    points_inside = np.column_stack((x, y, z))
    return points_inside


def data_train_sphere():
    t = torch.linspace(0, T, t_N).view(-1, 1).float()
    x_boundary, y_boundary, z_boundary = generate_sphere_boundary_points(num_boundary_points)
    boundary_points = np.column_stack((x_boundary, y_boundary, z_boundary))
    points_inside = generate_interior_points_sphere(radius=0.5, n_points=num_interior_points)

    return t, points_inside, boundary_points

def data_test_sphere():
    t = torch.linspace(0, T, test_N).view(-1, 1).float()
    x_boundary, y_boundary, z_boundary = generate_sphere_boundary_points(num_boundary_points, lhs_sampling=False)
    boundary_points = np.column_stack((x_boundary, y_boundary, z_boundary))
    points_inside = generate_interior_points_sphere(radius=0.5, n_points=num_test_points, lhs_sampling=False)
    xyz = torch.from_numpy(np.vstack((points_inside, boundary_points)))
    xyz_test = xyz.repeat(len(t), 1)
    t_vector_test = t.repeat_interleave(len(xyz)).view(-1, 1)
    txyz_test = torch.cat((t_vector_test, xyz_test), dim=1).float().to(device)

    with torch.no_grad():
        txyz_test_exact = exact_u(txyz_test)

    return xyz, txyz_test, txyz_test_exact


#Neural Network
class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        
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
    def __init__(self, net, xyz, xyz_bounds, t, txyz_test, txyz_test_exact, ha, beta):
        self.net = net
        self.ha = ha
        self.beta = beta

        self.xyz_bounds = torch.from_numpy(xyz_bounds).to(device, dtype=torch.float32)
        self.x_N = len(xyz_bounds)
        self.xyz = torch.from_numpy(xyz).to(device, dtype=torch.float32)
        self.xyz_N = len(xyz)

        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        
        self.t = t.to(device, dtype=torch.float32)
        self.t_N = len(t)
        self.dt = ((ub[0] - lb[0]) / (self.t_N - 1))

        self.txyz_test = txyz_test.to(device, dtype=torch.float32)
        self.txyz_test_exact = txyz_test_exact.to(device, dtype=torch.float32)

        self.coef = self.coef_d(np.arange(self.t_N))
        self.loss_records = {key: [] for key in ['initial', 'boundary', 'pde', 'total', 'error', 'predictions']}
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
        def create_full_txyz(t, xyz, N):
            t_repeated = torch.repeat_interleave(t[:, 0], N).view(-1, 1).to(device)
            xyz_repeated = xyz.repeat(len(t), 1)
            return torch.cat((t_repeated, xyz_repeated), dim=1)

        self.xyzb = torch.cat((self.xyz, self.xyz_bounds), dim=0)
        self.txyz_t0 = torch.cat((torch.full_like(torch.zeros(self.xyz_N + self.x_N, 1), self.t[0][0], device=device), self.xyzb), dim=1)
        self.txyz = create_full_txyz(self.t, self.xyzb, self.xyz_N + self.x_N)

        # Boundary conditions
        self.txyz_bounds = create_full_txyz(self.t, self.xyz_bounds, self.x_N)


        self.u_bounds = exact_u(self.txyz_bounds)
        if noise == 0.0:
            self.u_t0 = exact_u(self.txyz_t0)
        else:
            self.u_t0 = exact_u(self.txyz_t0)
            sigma = (noise * self.u_t0.cpu().detach().numpy())**2
            mu = 0 
            s = np.random.normal(mu, sigma, self.u_t0.shape)
            self.u_t0 += torch.from_numpy(s).cuda()

        self.F_txyz = F(self.txyz)


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def train_U(self, x):
        H = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(H)

    def PDE_loss(self):
        coef = self.coef
        x = Variable(self.txyz, requires_grad=True)

        u_pred = self.train_U(x).to(device)
        u_x_y_z = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_x, u_y, u_z = u_x_y_z[:, [1]], u_x_y_z[:, [2]], u_x_y_z[:, [3]]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, [1]]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, [2]]
        u_zz = torch.autograd.grad(u_z, x, grad_outputs=torch.ones_like(u_z), create_graph=True)[0][:, [3]]

        u_n = u_pred.reshape(self.t_N, -1)
        Lu = 1.0*(u_xx.reshape(self.t_N, -1) + u_yy.reshape(self.t_N, -1) + u_zz.reshape(self.t_N, -1)) - 1.0 *(u_x.reshape(self.t_N, -1) + u_y.reshape(self.t_N, -1) + u_z.reshape(self.t_N, -1))
        F_n = self.F_txyz.reshape(self.t_N, -1)

        loss = torch.tensor(0.0, device=device)

        for n in range(1, self.t_N):
            pre_Ui = ((Lu[n] + F_n[n]) / coef[0] + (coef[n-1] / coef[0]) * u_n[0]).to(device)
            if n > 1:
                for k in range(1, n):
                    pre_Ui += ((coef[n-k-1] - coef[n-k]) / coef[0]) * u_n[k]
            loss += torch.mean((pre_Ui - u_n[n]) ** 2)

        return loss

    def compute_loss(self):
        # Initial condition loss
        u_pred_t0 = self.train_U(self.txyz_t0)
        loss_initial = torch.mean((u_pred_t0 - self.u_t0) ** 2)
        self.loss_records['initial'].append([self.net.iter, loss_initial.item()])

        # Boundary condition losses
        u_pred = self.train_U(self.txyz_bounds)
        loss_boundary = torch.mean((u_pred - self.u_bounds) ** 2)
        self.loss_records['boundary'].append([self.net.iter, loss_boundary.item()])

        # PDE loss
        loss_pde = self.PDE_loss()
        self.loss_records['pde'].append([self.net.iter, loss_pde.item()])

        return loss_initial, loss_boundary, loss_pde

    def LBGFS_loss(self):
        self.optimizer_LBGFS.zero_grad()
        loss_initial, loss_boundary, loss_pde = self.compute_loss()
        total_loss = loss_initial + loss_boundary + loss_pde
        self.loss_records['total'].append([self.net.iter, total_loss.item()])
        total_loss.backward()

        if self.net.iter % 100 == 0:
            print('Iter:', self.net.iter, 'Loss:', total_loss.item())

        self.net.iter += 1
        
        with torch.no_grad():
            predictions = self.train_U(self.txyz_test).cpu().numpy()
            exact_values = self.txyz_test_exact.cpu().numpy()
            error = np.linalg.norm(predictions - exact_values, 2) / np.linalg.norm(exact_values, 2)
            self.loss_records['error'].append([self.net.iter, error])

            if self.net.iter % 10 == 0:
                predictions = self.train_U(self.txyz_test).cpu().numpy()
                self.loss_records['predictions'].append(predictions.tolist())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return total_loss

    def train_adam(self, adam_epochs):
        self.max_iterations = adam_epochs
        self.optimizer_adam = optim.Adam(self.net.parameters(), lr=1e-3)
        
        for epoch in range(adam_epochs):
            self.optimizer_adam.zero_grad()
            loss_initial, loss_boundary, loss_pde = self.compute_loss()
            total_loss = loss_initial + loss_boundary + loss_pde
            total_loss.backward()
            self.optimizer_adam.step()
            
            if epoch % 100 == 0:
                print(f'Adam Epoch {epoch}: Loss = {total_loss.item()}')

            with torch.no_grad():
                predictions = self.train_U(self.txyz_test).cpu().numpy()
                exact_values = self.txyz_test_exact.cpu().numpy()
                error = np.linalg.norm(predictions - exact_values, 2) / np.linalg.norm(exact_values, 2)
                self.loss_records['error'].append([self.net.iter, error])

                if epoch % 10 == 0:
                    predictions = self.train_U(self.txyz_test).cpu().numpy()
                    self.loss_records['predictions'].append(predictions.tolist())

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
        pred = self.train_U(self.txyz_test).cpu().numpy()
        exact = self.txyz_test_exact.cpu().numpy()
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
    layers  = [4] + [n_neurons] * n_layers + [1]
    net = Net(layers).to(device)
    torch.nn.DataParallel(net)

    # Distributed-order parameters
    beta = 1
    ha = 1/100
    T = 1

    lb = np.array([0.0, -0.5, -0.5, -0.5]) # low boundary
    ub = np.array([1.0, 0.5, 0.5, 0.5]) # up boundary
    
    '''train data'''
    t_N = 80
    num_boundary_points = 100
    num_interior_points = 300  

    x, y, z = define_sphere_boundary()
    t, xyz, xyz_bounds = data_train_sphere()
    
    '''test data'''
    test_N = 100
    num_boundary_points = 150
    num_test_points = 8
    xyz_test, txyz_test, txyz_test_exact = data_test_sphere()


    '''Train'''
    model = Model(
            net=net,
            xyz=xyz,
            xyz_bounds=xyz_bounds,
            t=t,
            txyz_test=txyz_test,
            txyz_test_exact=txyz_test_exact,
            ha=ha,
            beta=beta
        )


    model.train(LBGFS_epochs=20000)
