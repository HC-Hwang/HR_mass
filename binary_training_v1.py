"""
Author: Hsiang-Chih Hwang
v0 Start date: 3/7/2023
v1 Start date: 7/19/2023

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm
import astropy.units as u
from astropy.table import Table, Column, MaskedColumn, join
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord

import math

from scipy import stats
from scipy import special
import numpy as np


torch_pi = torch.tensor(math.pi)

from matplotlib import rcParams
rcParams.update({'font.size': '18'})
rcParams.update({'font.family': 'sans-serif'})
rcParams.update({'axes.facecolor': 'white'})
rcParams.update({'axes.edgecolor': 'black'})
rcParams.update({'axes.labelcolor': 'black'})
rcParams.update({'xtick.top': 'True'})
rcParams.update({'xtick.major.pad': '6.0'})
rcParams.update({'xtick.color': 'k'})
rcParams.update({'xtick.direction': 'in'})
rcParams.update({'ytick.right': 'True'})
rcParams.update({'ytick.major.pad': '6.0'})
rcParams.update({'ytick.color': 'k'})
rcParams.update({'ytick.direction': 'in'})
rcParams.update({'legend.fontsize': '18'})
rcParams.update({'figure.figsize': '8.0, 6.0'})
rcParams.update({'figure.dpi': '100'})
rcParams.update({'figure.facecolor': 'white'})
rcParams.update({'figure.edgecolor': 'white'})
rcParams.update({'image.cmap': 'rainbow'})

class BinaryTraining():
    def __init__(self, binary_table=None, sample_type='', sample_file_name=''):
        
        if binary_table is None:
            print('For model-loading only. No binary table is loaded. Some functions may not be working.')
            return
            
        
        self.wb_table = binary_table
        self.sample_file_name = sample_file_name

        print('torch version:', torch.__version__)
                
                
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print('self.device: ', self.device)
        
        #self.outlier_gaussian_norm = torch.tensor(1. - 0.5 * (1 + special.erf(-30/15 / np.sqrt(2.))))
        
        if sample_type in ['Gaia', 'Mock']:
            self.sample_type = sample_type
        else:
            print('Error: no support sample type (should be Gaia or Mock)')
            
        if self.sample_type == 'Gaia':
                        
            self.wb_table['DeltaG'] = self.wb_table['phot_g_mean_mag2'] - self.wb_table['phot_g_mean_mag1']
            
            self.wb_table['sep_arcsec'] = self.wb_table['pairdistance'] * 3600.
            
            self.wb_table['absg1'] = self.wb_table['phot_g_mean_mag1'] + 5 * np.log10(self.wb_table['parallax1']) - 10.
            self.wb_table['absg2'] = self.wb_table['phot_g_mean_mag2'] + 5 * np.log10(self.wb_table['parallax1']) - 10.
            
            self.wb_table['v'] = 4.7 * np.sqrt((self.wb_table['pmra2']-self.wb_table['pmra1'])**2 + 
                                          (self.wb_table['pmdec2']-self.wb_table['pmdec1'])**2
                                         ) / self.wb_table['parallax1']
            
            self.wb_table['u'] = self.wb_table['v'] * np.sqrt(self.wb_table['sep_AU'])
            
            
            self.wb_table['u_sigma'] = self.wb_table['u'] / self.wb_table['dpm_over_error']
            
            #self.wb_table['u_sigma'] = self.wb_table['u'].data / self.wb_table['dpm_over_error']
            #self.u_sigma = (self.wb_table['u'].data / self.wb_table['dpm_over_error']).astype(np.float32)
            
        self.wb_table['norm_factor'] = 0.5 * (1. - special.erf((-self.wb_table['u']) / self.wb_table['u_sigma'] / np.sqrt(2)))

    


    def func_pu_8(self, tilde_u, A=torch.tensor(4.95e-3), B=torch.tensor(2.24e-3), C=torch.tensor(3.85), u0=torch.tensor(36.09)):
        #version 8 of p(u)
    
        return A * tilde_u * torch.exp( -1 * (
            B * tilde_u**2 + torch.exp((tilde_u - u0) / C)
        ))



    def gaussian(self, x, x0, sigma, norm_factor=None):
    
        if norm_factor is None:
            return 1. / sigma / torch.sqrt(2. * torch_pi) * torch.exp(-0.5 * (x - x0)**2 / sigma**2)
        else:
            return 1. / norm_factor / sigma / torch.sqrt(2. * torch_pi) * torch.exp(-0.5 * (x - x0)**2 / sigma**2)


    def outlier_gaussian(self, tilde_u, outlier_u0=torch.tensor(30.), outlier_sigma=torch.tensor(15.)):
        return 1. / self.outlier_gaussian_norm * self.gaussian(tilde_u, outlier_u0, outlier_sigma)



    def loss_fn_4_func_pu_Gaussian_outlier(self, expm1_predict, expm2_predict, u, u_sigma=None, norm_factor=None, 
                                       int_umax = 80., int_du=0.01, outlier_u0=torch.tensor(30.), outlier_sigma=torch.tensor(15.)
                                       ):
    
        m_epsilon = torch.tensor(1e-10)
        mtot = torch.exp(expm1_predict) + torch.exp(expm2_predict) + m_epsilon
        sqrt_mtot = torch.sqrt(mtot)
    
        p_epsilon = torch.tensor(1e-10)
    
        if norm_factor is None:
            norm_factor = torch.ones_like(u).to(self.device)

    
        if u_sigma is None: #no errors are provided. Assume no errors
    
            log_prob = torch.log(
                self.f_good * 1. / sqrt_mtot * self.func_pu_8(u / sqrt_mtot) +
                (1. - self.f_good) * self.outlier_gaussian(u/sqrt_mtot, outlier_u0, outlier_sigma) +
                p_epsilon
            )
        else: #integrate over tilde u, assuming the uncertainty distribution is a Gaussian
    
    
            int_ulist = torch.arange(0., int_umax, int_du).to(self.device)
    
            uncertainty_gaussian = self.gaussian(int_ulist[None, :], (u/sqrt_mtot)[:, None], (u_sigma/sqrt_mtot)[:, None])
    
            log_prob = torch.log( torch.sum(
                (
                    self.f_good * 1. / sqrt_mtot[:, None] * self.func_pu_8(int_ulist)[None, :] +
                    (1. - self.f_good) * self.outlier_gaussian(int_ulist, outlier_u0, outlier_sigma)[None, :] 
                ) *  uncertainty_gaussian * int_du + p_epsilon / (int_umax / int_du)
                , axis=1
                ) / norm_factor + p_epsilon
            )
    
        return -torch.mean(log_prob)

    def sample_preparation(self, train_data_frac, batch_size=1024):
        
        self.train_data_frac = train_data_frac
        self.batch_size = batch_size
        
        self.bprp_1 = (self.wb_table['bp_rp1'].data).astype(np.float32)
        self.absg_1 = (self.wb_table['absg1'].data).astype(np.float32)
        self.bprp_2 = (self.wb_table['bp_rp2'].data).astype(np.float32)
        self.absg_2 = (self.wb_table['absg2'].data).astype(np.float32)
    
        self.u = (self.wb_table['u'].data).astype(np.float32)
        self.u_sigma = self.wb_table['u_sigma'].astype(np.float32)
        self.norm_factor = (self.wb_table['norm_factor'].data).astype(np.float32)
    
    
        features_1 = np.dstack([self.bprp_1, self.absg_1])
        features_1 = features_1[0, :, :]
        features_2 = np.dstack([self.bprp_2, self.absg_2])
        features_2 = features_2[0, :, :]
    
    
        n_data = len(self.wb_table)
    
        n_train = int(n_data * train_data_frac)
        print('n_train: ', n_train)
    
        s_train = np.zeros(n_data, dtype=np.bool)
        
        s_train[np.random.choice(range(n_data), size=n_train, replace=False)] = True
    
        s_test = ~s_train
        n_test = np.sum(s_test)
    
        print('n_test: ', n_test)
    
    
        #set up train features in batch
        self.train_features_1 = torch.tensor(features_1[s_train], dtype=torch.float32).to(self.device)
        self.train_features_2 = torch.tensor(features_2[s_train], dtype=torch.float32).to(self.device)
    
    
        self.train_features_1 = self.train_features_1.reshape(n_train, 2)
        self.train_features_2 = self.train_features_2.reshape(n_train, 2)
    
    
        self.train_u = torch.tensor(self.u[s_train], dtype=torch.float32).to(self.device)
        self.train_u = self.train_u.reshape(n_train, 1)
    
        self.train_usigma = torch.tensor(self.u_sigma[s_train], dtype=torch.float32).to(self.device)
        self.train_usigma = self.train_usigma.reshape(n_train, 1)
    
        self.train_norm_factor = torch.tensor(self.norm_factor[s_train], dtype=torch.float32).to(self.device)
        self.train_norm_factor = self.train_norm_factor.reshape(n_train, 1)
    
        self.train_dataset = list(zip(self.train_features_1, self.train_features_2, self.train_u, self.train_usigma, self.train_norm_factor))
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
    
    
        #set up test features. Single set, no multiple batches
    
        self.test_features_1 = torch.tensor(features_1[s_test], dtype=torch.float32).to(self.device)
        self.test_features_2 = torch.tensor(features_2[s_test], dtype=torch.float32).to(self.device)
        self.test_features_1 = self.test_features_1.reshape(n_test, 2)
        self.test_features_2 = self.test_features_2.reshape(n_test, 2)
        self.test_u = torch.tensor(self.u[s_test], dtype=torch.float32).to(self.device)
        self.test_u = self.test_u.reshape(n_test, 1)
        self.test_usigma = torch.tensor(self.u_sigma[s_test], dtype=torch.float32).to(self.device)
        self.test_usigma = self.test_usigma.reshape(n_test, 1)
        self.test_norm_factor = torch.tensor(self.norm_factor[s_test], dtype=torch.float32).to(self.device)
        self.test_norm_factor = self.test_norm_factor.reshape(n_test, 1)
    
        # test_data = [test_features_1, test_features_2, test_labels, test_usigma]
    
        self.test_data = {
            'features_1': self.test_features_1, 
            'features_2': self.test_features_2,
            'test_u': self.test_u,
            'test_usigma': self.test_usigma,
            'test_norm_factor': self.test_norm_factor
        }
    
        #return train_loader, test_data
        
    def init_model_parameters(self, num_epochs, lr,
        f_outlier, dropout_frac=0.2,
        num_inputs=2, num_hiddens=128, num_outputs=1,
        int_umax=100., int_du=0.01,
        outlier_u0=30, outlier_sigma=20,
        output_png=False, result_path='./', random_seed=None
    ):
        self.num_epochs = num_epochs
        self.lr = lr
        self.f_outlier = f_outlier
        self.f_good = torch.tensor(1. - f_outlier)
        self.dropout_frac = dropout_frac
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        
        self.output_png = output_png
        self.result_path = result_path
        
        self.int_umax = int_umax
        self.int_du= int_du
        
        self.outlier_u0 = torch.tensor(outlier_u0)
        self.outlier_sigma = torch.tensor(outlier_sigma)
        self.outlier_gaussian_norm = torch.tensor(1. - 0.5 * (1 + special.erf(-self.outlier_u0/ self.outlier_sigma / np.sqrt(2.))))
        
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)
        
    def training(self):
        
        # training losses
        self.train_ls, self.test_ls = [], []
        
    
        self.model = mmodel_2(self.num_inputs, self.num_hiddens, self.num_outputs, self.dropout_frac)
    
        if torch.cuda.is_available():
            self.model.cuda()
    
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        if self.output_png: #prepare dataset to make plots
            
            if len(self.wb_table) <= 10000:
                plot_random_idx = np.arange(10000)
            else:
                plot_random_idx = np.random.choice(len(self.wb_table), 10000, replace=False)
                
            self.data1 = np.dstack([self.wb_table['bp_rp1'][plot_random_idx], self.wb_table['absg1'][plot_random_idx]])
            self.data1 = (self.data1[0, :, :]).astype(np.float32)
            self.data1 = torch.tensor(self.data1).to(self.device)
        
            self.data2 = np.dstack([self.wb_table['bp_rp2'][plot_random_idx], self.wb_table['absg2'][plot_random_idx]])
            self.data2 = (self.data2[0, :, :]).astype(np.float32)
            self.data2 = torch.tensor(self.data2).to(self.device)
            
            # self.data1 = np.dstack([self.wb_table['bp_rp1'], self.wb_table['absg1']])
            # self.data1 = (self.data1[0, 0:10000, 0:10000]).astype(np.float32)
            # self.data1 = torch.tensor(self.data1).to(self.device)
        
            # self.data2 = np.dstack([self.wb_table['bp_rp2'], self.wb_table['absg2']])
            # self.data2 = (self.data2[0, 0:10000, 0:10000]).astype(np.float32)
            # self.data2 = torch.tensor(self.data2).to(self.device)
        
    
        
        for e in range(int(self.num_epochs)):
    
            print(e)
    
            if self.output_png:
                vmax = 2.5
                vmin = 0.2
    
                data1_mass = torch.exp(self.model(self.data1)).detach().cpu().flatten()
                data2_mass = torch.exp(self.model(self.data2)).detach().cpu().flatten()
    
                plt.scatter(
                    self.data1[:, 0].cpu(),
                    self.data1[:, 1].cpu(),
                    c=data1_mass,
                    s=10, marker='s', vmax=vmax, vmin=vmin, label=None
                )
                plt.scatter(
                    self.data2[:, 0].cpu(),
                    self.data2[:, 1].cpu(),
                    c=data2_mass,
                    s=10, marker='s', vmax=vmax, vmin=vmin, label=None
                )
    
                #plt.colorbar(label=r'Ln(Mass (M$_\odot$))')
                plt.colorbar(label=r'Measured mass (M$_\odot$)')
                #plt.legend()
                plt.gca().invert_yaxis()
                plt.xlabel('BP-RP')
                plt.ylabel(r'$M_G$')
                plt.title('Epoch: %d' %(e))
    
                plt.savefig(self.result_path + '_%05d.png' %(e), dpi=50)
                plt.close()
    
    
            counter = 0
            loss_average = 0
    
            # the dataloader will do the shuffling and batching for us
            for (batch_features_1, batch_features_2, batch_u, batch_usigma, batch_norm_factor) in self.train_loader:

                expm1_predict = self.model(batch_features_1)
                expm2_predict = self.model(batch_features_2)
    
                #with normalization factors for low-SNR u
                loss = self.loss_fn_4_func_pu_Gaussian_outlier(expm1_predict.flatten(), expm2_predict.flatten(), batch_u.flatten(),  batch_usigma.flatten(), norm_factor=batch_norm_factor.flatten(), int_umax=self.int_umax, int_du=self.int_du)
    
                if torch.isnan(loss):
                    
                    break
    
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
    
                # accumulate the losses from individual batches
                counter += 1
                loss_average += loss
    
            
            if torch.isnan(loss):
                print('nan loss break')
                break
            
            if torch.cuda.is_available():
                loss_average = loss_average.cpu()
    
            loss_average = loss_average.detach().numpy()/counter
            self.train_ls.append(loss_average)
    
            expm1_predict = self.model(self.test_data['features_1'])
            expm2_predict = self.model(self.test_data['features_2'])
            
    
            #normalization factor for low-snr u
            loss_test = self.loss_fn_4_func_pu_Gaussian_outlier(expm1_predict.flatten(), expm2_predict.flatten(), self.test_data['test_u'].flatten(), self.test_data['test_usigma'].flatten(), norm_factor=self.test_data['test_norm_factor'].flatten(), int_umax=self.int_umax, int_du=self.int_du)
            
            #loss_test = torch.tensor(0.)
    
            if torch.cuda.is_available():
                loss_test = loss_test.cpu()
    
            self.test_ls.append(loss_test.detach().numpy())
    
            if e % 1 == 0:
                print('iter %s:' % e, 'training loss = %.5f' % loss_average,\
                      'validation loss = %.5f' % loss_test)   
                
                
        torch.save(self.model.state_dict(), self.result_path + 'trained_model.pth')
            
    
    def predict_mass(self, bprp, absg):
        data = np.dstack([bprp, absg])
        data = (data[0, :, :]).astype(np.float32)
        data = torch.tensor(data).to(self.device)
        mass = torch.exp(self.model(data)).detach().cpu().flatten()
        return data, mass
        
            
            
    def plot_training_result(self, predict_mass_type='sampling_median', vmin=0.2, vmax=2.5, label='', marker='.'):
        
        if predict_mass_type == 'sampling_median' and 'predict_mass1_median' in self.wb_table.columns: #already computed
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            s1 = (
                self.wb_table['predict_mass1_error'] / self.wb_table['predict_mass1_median'] < 0.5
            )
            s2 = (
                self.wb_table['predict_mass2_error'] / self.wb_table['predict_mass2_median'] < 0.5
            )
            
            marker_size = 20
            plt.scatter(
                        self.wb_table['bp_rp1'][s1],
                        self.wb_table['absg1'][s1],
                        c=self.wb_table['predict_mass1_median'][s1],
                        s=marker_size, marker='.', vmax=vmax, vmin=vmin, cmap='rainbow'
            )
            
            plt.scatter(
                        self.wb_table['bp_rp2'][s2],
                        self.wb_table['absg2'][s2],
                        c=self.wb_table['predict_mass2_median'][s2],
                        s=marker_size, marker='.', vmax=vmax, vmin=vmin, cmap='rainbow'
            )
            plt.colorbar(label=r'Inferred mass (M$_\odot$)')
            #plt.legend()
            plt.gca().invert_yaxis()
            plt.xlabel(r'BP$-$RP')
            plt.ylabel(r'$\rm{M_G}$')
            
            
            return
        
        
        if predict_mass_type == 'single':
        
            data1, mass1 = self.predict_mass(self.wb_table['bp_rp1'], self.wb_table['absg1'])
            data2, mass2 = self.predict_mass(self.wb_table['bp_rp2'], self.wb_table['absg2'])
        elif predict_mass_type == 'sampling_median':
            data1, mass1 = self.predict_mass(self.wb_table['bp_rp1'], self.wb_table['absg1'])
            data2, mass2 = self.predict_mass(self.wb_table['bp_rp2'], self.wb_table['absg2'])
            mass1 = self.wb_table['predict_mass1_median']
            mass2 = self.wb_table['predict_mass2_median']
        else:
            print('predict_mass_type needs to be single or sampling_median')
        
        plt.scatter(
            data1[:, 0].cpu(),
            data1[:, 1].cpu(),
            c=mass1,
            s=10, marker=marker, vmax=vmax, vmin=vmin, label=label
        )
        
        
        plt.scatter(
            data2[:, 0].cpu(),
            data2[:, 1].cpu(),
            c=mass2,
            s=10, marker=marker, vmax=vmax, vmin=vmin, label=None
        )
        

        plt.colorbar(label=r'Mass (M$_\odot$)')
        plt.legend()

        
        plt.gca().invert_yaxis()
        plt.xlabel('BP-RP')
        plt.ylabel(r'$M_G$')
        
        return
        
        #plt.show()
        
    def plot_p_tilde_u(self, hist_du=0.5, hist_max=100):
        
        data1, mass1 = self.predict_mass(self.wb_table['bp_rp1'], self.wb_table['absg1'])
        data2, mass2 = self.predict_mass(self.wb_table['bp_rp2'], self.wb_table['absg2'])

        plt.hist(
            self.wb_table['u'] / np.sqrt(mass1 + mass2),
            density=1,
            bins=np.arange(0., hist_max, hist_du),
            histtype='step', label='Training result'
        )
        
        # print(-torch.mean(torch.log(
        #     self.f_good * func_pu_8(self.wb_table['u'] / np.sqrt(mass1 + mass2 + 1e-5)) +
        #     (1. - self.f_good) * self.outlier_gaussian(self.wb_table['u'] / np.sqrt(mass1 + mass2 + 1e-5), outlier_u0=30, outlier_sigma=15) + 
        #     1e-5
        #     )))
        
        torch_u_list = torch.arange(0., 100., 0.001)
        plt.plot(
            torch_u_list,
            self.f_good * self.func_pu_8(torch_u_list)
        )
        
        plt.plot(
            torch_u_list,
            (1. - self.f_good) * self.outlier_gaussian(torch_u_list, outlier_u0=self.outlier_u0, outlier_sigma=self.outlier_sigma)
        )
        
        
        plt.plot(
            torch_u_list,
            self.f_good * self.func_pu_8(torch_u_list) + (1. - self.f_good) * self.outlier_gaussian(torch_u_list, outlier_u0=self.outlier_u0, outlier_sigma=self.outlier_sigma),
            c='k'
        )
        
        
        plt.xlabel(r'$\tilde{u}$')
        plt.ylabel(r'$p(\tilde{u})$')
        plt.legend(fontsize=15)
        plt.show()
        
    def derive_predict_median_mass_from_sampling(self, N_sample=1000):
        
        mass1_list = []
        mass2_list = []
        for i in tqdm(range(N_sample)):
            data1, mass1 = self.predict_mass(self.wb_table['bp_rp1'], self.wb_table['absg1'])
            data2, mass2 = self.predict_mass(self.wb_table['bp_rp2'], self.wb_table['absg2'])
            
            mass1_list.append(mass1.numpy())
            mass2_list.append(mass2.numpy())
        
        mass1_list = np.array(mass1_list)
        mass2_list = np.array(mass2_list)
        self.wb_table['predict_mass1_median'] = np.median(mass1_list, axis=0)
        self.wb_table['predict_mass2_median'] = np.median(mass2_list, axis=0)
        
        self.wb_table['predict_mass1_error'] = np.std(mass1_list, axis=0)
        self.wb_table['predict_mass2_error'] = np.std(mass2_list, axis=0)
        


class mmodel_2(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, dropout_frac):
        super(mmodel_2, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, num_hiddens), # wx + b
            nn.Dropout(p=dropout_frac),
            torch.nn.GELU(), # apply activation function
            torch.nn.Linear(num_hiddens, num_hiddens),
            nn.Dropout(p=dropout_frac),
            torch.nn.GELU(),
            torch.nn.Linear(num_hiddens, num_hiddens),
            nn.Dropout(p=dropout_frac),
            torch.nn.GELU(),
            torch.nn.Linear(num_hiddens, num_hiddens),
            nn.Dropout(p=dropout_frac),
            torch.nn.GELU(),
            torch.nn.Linear(num_hiddens, num_hiddens),
            nn.Dropout(p=dropout_frac),
            torch.nn.GELU(),
            torch.nn.Linear(num_hiddens, num_outputs),
        )
        
    def forward(self, x):
        return self.mlp(x)
    

print('v72')