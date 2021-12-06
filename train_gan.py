import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pickle
import time
import os
import json
from tqdm import tqdm
import pathlib

from data_loader import *
from model import *


def train():
    
    logs = {

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_features' : 7, ### change according to your dataset
    'z_dim' : 64,
    'hidden_shape': 128,
    'epoch' : 1000,
    'batch_size' : 64,
    'seq_len' : 24,
    'clipping_rate' : 0.01,
    'n_critic' : 5,
    'num_layers': 6,
    'dataset_name' : 'mimiciii',
    'loss_mode': 'lsgan',
    'architecture': 'Linear'

}
    
    device = logs['device']

    n_features = logs['n_features']

    z_dim = logs['z_dim']

    hidden_shape = logs['hidden_shape']

    epoch = logs['epoch']

    batch_size = logs['batch_size']

    seq_len = logs['seq_len']

    clipping_rate = logs['clipping_rate']

    n_critic = logs['n_critic']

    num_layers = logs['num_layers']

    dataset_name = logs['dataset_name'] # choice == 'sine' / 'google'

    loss_mode = logs['loss_mode'] # choice == 'wgan' / 'lsgan' / 'normal'

    architecture = logs['architecture'] # choice == 'LSTM' / 'cnn' / 'GRU'

    ################################
    
    file_name = f'{architecture}-{dataset_name}-{loss_mode}'

    folder_name = f'final_saved_models/{time.time():.4f}-{file_name}'
    
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    
    pathlib.Path(f'{folder_name}/output/gan').mkdir(parents=True, exist_ok=True) 
    
    with open(f'{folder_name}/params.json', 'w') as f:
        
        json.dump(logs, f)
        
        f.close()
        
    writer = SummaryWriter(log_dir = folder_name, comment = f'{file_name}', flush_secs = 45)

    
    ##############################
    
    data_dir = 'data'



    if seq_len == 24:

        if dataset_name == 'google':

            if os.path.exists(f'{data_dir}/train_data_google_seq_24.pkl') and os.path.exists(f'{data_dir}/test_data_google_seq_24.pkl'):

                print(f'Loading Saved data')

                with open(f'{data_dir}/train_data_google_seq_24.pkl', 'rb') as f:

                    train_data = pickle.load(f)

                    f.close()

                    print(f'Google Data loaded with sequence 24')


        elif dataset_name == 'sine5-small':

            if os.path.exists(f'{data_dir}/train_data_sine5_small_seq_24.pkl') and os.path.exists(f'{data_dir}/test_data_sine5_small_seq_24.pkl'):

                print(f'Loading Saved data')

                with open(f'{data_dir}/train_data_sine5_small_seq_24.pkl', 'rb') as f:

                    train_data = pickle.load(f)

                    f.close()

                    print(f'Sine Small Data with 5 dimension loaded with sequence 24')
                    
        elif dataset_name == 'mimiciii':

            if os.path.exists(f'{data_dir}/synthetic_data_numeric.csv'):

                print(f'Loading Saved data')
                
                train_data = SyntheticDataLoader(f'{data_dir}/synthetic_data_numeric.csv')

                print(f'Synthetic Mimic Data Loaded!')
                

    data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size)

    print(f'Data loading Complete')
    
    #####################
    
    
    if architecture == 'cnn':

        generator = SimpleConvGen(z_dim).to(device)

        discriminator = SimpleConvDisc(n_features).to(device)

    elif architecture == 'GRU' or architecture == 'LSTM':

        generator = CommonGRU(z_dim, hidden_shape, hidden_shape, num_layers, architecture, activation_fn = nn.Tanh()).to(device)

        discriminator = CommonGRU(n_features, hidden_shape, 1, num_layers, architecture, activation_fn = None).to(device)
        
    elif architecture == 'Linear':

        generator = LinearGen(z_dim, n_features, hidden_shape).to(device)
        
        discriminator = LinearDisc(n_features, hidden_shape).to(device)
        
        
    ######################################
    
    gen_optim = torch.optim.Adam(generator.parameters(), lr = 0.0002,  betas = (0.9, 0.999), weight_decay = 0.0001)
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr = 0.0002, betas = (0.9, 0.999), weight_decay = 0.0001)
    
    
    ae_loss = []
    gen_loss = []
    disc_loss = []

    ae_criterion = nn.MSELoss()

    gan_criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    one = torch.tensor([1], dtype = torch.float, device = device)

    mone = one * -1
    
    if architecture == 'Linear':

        fixed_noise = torch.randn(batch_size, z_dim, dtype = torch.float, device = device)
        
    else:
        
        fixed_noise = torch.randn(batch_size, seq_len, z_dim, dtype = torch.float, device = device)
        

    gen_iterations = 0
    
    
    start_time = time.time()

    for running_gan_epoch in tqdm(range(epoch), ascii = True, colour = 'blue'):

        ### GAN Training

        for i, data in enumerate(data_loader):

            ### Training Discriminator

            data = data.to(device)

            batch_size = data.size(0)
            
            if architecture == 'Linear':

                z = torch.randn(batch_size, z_dim, dtype = torch.float, device = device)
                
            else:
                
                z = torch.randn(batch_size, seq_len, z_dim, dtype = torch.float, device = device)
                
                
            if loss_mode == 'normal':

                discriminator.zero_grad()

                real_output = discriminator(data).view(-1)

                fake_output = discriminator(generator(z).detach()).view(-1) 
                
                if architecture == 'Linear':

                    label = torch.full((batch_size,), real_label, dtype = torch.float, device = device)
                    
                else:
                    
                    label = torch.full((batch_size * seq_len,), real_label, dtype = torch.float, device = device)
                    
                error_real = gan_criterion(real_output, label)

                error_real.backward()

                label.fill_(fake_label)

                error_fake = gan_criterion(fake_output, label)

                error_fake.backward()

                d_loss = error_real + error_fake

                disc_optim.step()

                disc_loss.append(d_loss.item())

                ### Generator Update

                generator.zero_grad()

                label.fill_(real_label)

                fake_output_gen = discriminator(generator(z)).view(-1) 

                g_loss = gan_criterion(fake_output_gen, label)

                g_loss.backward()

                gen_optim.step()

                gen_loss.append(g_loss.item())
                
                
            elif loss_mode == 'wgan':
                
                for p in discriminator.parameters():
                    p.required_grad = True

                discriminator.zero_grad()

                real_output = discriminator(data).view(-1)
                
                fake_output = discriminator(generator(z).detach()).view(-1)

                d_loss = -(torch.mean(real_output) - torch.mean(fake_output))

                d_loss.backward()

                disc_optim.step()
                
                disc_loss.append(d_loss.item())

                for p in discriminator.parameters():

                    p.data.clamp_(-clipping_rate, clipping_rate)

                if i%n_critic == 0:

                    for p in discriminator.parameters():

                        p.required_grad = False

                    generator.zero_grad()

                    fake_out = discriminator(generator(z)).view(-1)

                    g_loss = -torch.mean(fake_out)

                    g_loss.backward()

                    gen_optim.step()
                    
                    gen_loss.append(g_loss.item())
                    
                    
            elif loss_mode == 'lsgan':
            
                discriminator.zero_grad()

                real_output = discriminator(data).view(-1)
                
                fake_output = discriminator(generator(z).detach()).view(-1) 

                d_loss = 0.5 * ((torch.mean(real_output) - 1)**2) + 0.5 * (torch.mean(fake_output)**2) # loss for discriminator

                d_loss.backward()

                disc_optim.step()

                disc_loss.append(d_loss.item())

                ## GEN training

                generator.zero_grad()

                fake_out = discriminator(generator(z)).view(-1)

                g_loss = 0.5 * ((torch.mean(fake_out) - 1)**2)

                g_loss.backward()

                gen_optim.step()

                gen_loss.append(g_loss.item())

            if i%len(data_loader)==0:

                print(f'GAN Epoch: [{running_gan_epoch+1}/{epoch}], g_loss: {g_loss.item():.4f}, d_loss: {d_loss.item():.4f}')
                writer.add_scalar('D_Loss', d_loss.item(), running_gan_epoch)
                writer.add_scalar('G_Loss', g_loss.item(), running_gan_epoch)
                
            if i%len(data_loader)==0 and running_gan_epoch%100==0:
                
                with torch.no_grad():
                    
                    fake = generator(fixed_noise).detach().cpu()
                    
                    fig = plt.figure(constrained_layout=True, figsize=(20,10))
                    
                    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
                    
                    
                    idx = np.random.permutation(32)[:5]
                    
                    ax = fig.add_subplot(spec[0,0])
                    ax.set_title(f'Original idx: {idx[0]}',
                         fontsize=20,
                         color='red',
                         pad=10)
                    ori_data = data.detach().cpu()
                    plt.plot(ori_data[idx[0]])
                    
                    ax2 = fig.add_subplot(spec[0,1])
                    ax2.set_title(f'Synthetic idx: {idx[0]}',
                                  fontsize=20,
                                  color='red',
                                  pad=10)
                    plt.plot(fake[idx[0]])
                    fig.suptitle(f'Synthetic vs Real data',
                                 fontsize=16,
                                 color='blue')

                    plt.savefig(f'{folder_name}/output/gan/output_epoch-{running_gan_epoch+1}.png')
                    plt.close()
                    
                torch.save({

                    'epoch': running_gan_epoch+1,
                    'generator_state_dict_gan': generator.state_dict(),
                    'discriminator_state_dict_gan': discriminator.state_dict(),
                    'gen_optim_state_dict': gen_optim.state_dict(),
                    'disc_optim_state_dict': disc_optim.state_dict(),

                    }, os.path.join(f'{folder_name}', f'{file_name}-ep-{running_gan_epoch+1}.pth'))
                

    print(f'training_done!')

    end_time = time.time()

    elapsed_time = (end_time - start_time)/60.

    print(f'Total time: {elapsed_time:.4f} min')
    
    torch.save({

            'epoch': running_gan_epoch+1,
            'generator_state_dict_gan': generator.state_dict(),
            'discriminator_state_dict_gan': discriminator.state_dict(),
            'gen_optim_state_dict': gen_optim.state_dict(),
            'disc_optim_state_dict': disc_optim.state_dict(),

            }, os.path.join(f'{folder_name}', f'{file_name}-ep-{running_gan_epoch+1}-final.pth'))

    print('Weights Saved!!')

    print('training done!')
    
    
if __name__ == "__main__":
    
    train()
    