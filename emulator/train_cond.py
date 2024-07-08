import util
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
import model_cond
from PIL import Image, ImageDraw
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
from collections import OrderedDict
import pathlib
import matplotlib.pyplot as pl
import h5py
import torch.nn.functional as F
import os
from kornia.filters import SpatialGradient
try:    
    from telegram.ext import ApplicationBuilder, CommandHandler
    import asyncio
    TELEGRAM_BOT = True
except:
    TELEGRAM_BOT = False

def merge_images(image_batch, size, labels=None):
    b, h, w = image_batch.shape    
    img = np.zeros((int(h*size[0]), int(w*size[1])))
    for idx in range(b):
        i = idx % size[1]
        j = idx // size[1]
        maxval = np.max(image_batch[idx, :, :])
        minval = np.min(image_batch[idx, :, :])
        img[j*h:j*h+h, i*w:i*w+w] = (image_batch[idx, :, :] - minval) / (maxval - minval)

    img_pil = Image.fromarray(np.uint8(pl.cm.gray(img)*255))
    I1 = ImageDraw.Draw(img_pil)
    n = len(labels)
    for i in range(n):
        I1.text((2, 1+h*i), labels[i], fill=(255,255,0))
    img = np.array(img_pil)

    return img


class TelegramBot(object):
    def __init__(self):
        self.token = os.environ['TELEGRAM_TOKEN']
        self.chat_id = os.environ['TELEGRAM_CHATID']
                
    async def sendmessage(self, text):        
        await self.application.bot.sendMessage(chat_id=self.chat_id, text=text)

    async def sendphoto(self, photo):
        await self.application.bot.sendPhoto(chat_id=self.chat_id, photo=photo)

    def stop_training(self, update, context):
        self.bot_active = False
        update.message.reply_text("Stopping training...")
        self.stop()
        sys.exit()

    def send_message(self, text):
        self.application = ApplicationBuilder().token(self.token).build()   
        self.application.add_handler(CommandHandler('stop', self.stop_training))
        asyncio.run(self.sendmessage(text))

    def send_photo(self, photo):
        self.application = ApplicationBuilder().token(self.token).build()   
        self.application.add_handler(CommandHandler('stop', self.stop_training))
        asyncio.run(self.sendphoto(photo))
        
    def stop(self):
        os.kill(os.getpid(), signal.SIGINT)

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, config):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()

        self.config = config
        
        training_file = self.config['training_file']
        print(f"Reading training data {training_file}")

        f = h5py.File(training_file, 'r')
        self.images = f['images']
        self.convolved = f['convolved']
        self.modes = f['modes']
        self.instrument = f['instrument']

        self.n_training = self.images.shape[0]
        
    def __getitem__(self, index):

        # Select image
        image = self.images[index, :].reshape((self.config['n_pixel'], self.config['n_pixel']))
        convolved = self.convolved[index, :].reshape((self.config['n_pixel'], self.config['n_pixel']))
        modes = self.modes[index, :]
        instrument = self.instrument[index]

        return image.astype('float32'), modes.astype('float32'), convolved.astype('float32'), instrument.astype(np.int64)

    def __len__(self):
        return self.n_training        

class Training(object):
    def __init__(self, config):

        self.config = config

        self.cuda = torch.cuda.is_available()
        self.gpu = self.config['gpu']
        self.smooth = self.config['smooth']
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")        

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = self.config['batch_size']
                
        kwargs = {'num_workers': 4, 'pin_memory': False} if self.cuda else {}        
                
        print('Instantiating model...')
        self.model = model_cond.UNetFiLM(config=self.config).to(self.device)
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        print('Reading dataset...')
        self.dataset = Dataset(config=config)
        
        self.validation_split = self.config['validation_split']
        idx = np.arange(self.dataset.n_training)
        
        self.train_index = idx[0:int((1-self.validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-self.validation_split)*self.dataset.n_training):]

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, drop_last=True, **kwargs)

        # Spatial gradient operator
        self.spatial_gradient = SpatialGradient(mode='sobel', order=1, normalized=True).to(self.device)

        if (TELEGRAM_BOT):
            self.bot = TelegramBot()

    def init_optimize(self):

        self.lr = self.config['lr']
        self.wd = self.config['wd']
        self.n_epochs = self.config['n_epochs']
        
        print('Learning rate : {0}'.format(self.lr))        
        
        p = pathlib.Path('weights/')
        p.mkdir(parents=True, exist_ok=True)

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S").replace(':', '_')
        self.out_name = 'weights/{0}'.format(current_time)

        # Copy model
        f = open(model_cond.__file__, 'r')
        self.config['model_code'] = f.readlines()
        f.close()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler, gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=0.2*self.lr)

        # Creates once at the beginning of training if half-precision is used
        if self.config['precision'] == 'half':
            print("Working in half-precision")
            self.use_amp = True
        else:
            self.use_amp = False
                
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = 1e100
        
        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):            
            loss = self.train(epoch)
            loss_val = self.test()

            self.loss.append(loss)
            self.loss_val.append(loss_val)

            self.scheduler.step()

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'loss': self.loss,
                'loss_val': self.loss_val,
                'optimizer': self.optimizer.state_dict(),
                'hyperparameters': self.config
            }

            if (loss_val < best_loss):
                print(f"Saving model {self.out_name}.best.pth")                
                best_loss = loss_val
                torch.save(checkpoint, f'{self.out_name}.best.pth')

            if (self.config['save_all_epochs']):
                torch.save(checkpoint, f'{self.out_name}.ep_{epoch}.pth')
            else:
                torch.save(checkpoint, f'{self.out_name}.pth')


    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (images, modes, convolved, instrument) in enumerate(t):
            images = images.to(self.device)
            modes = modes.to(self.device)
            convolved = convolved.to(self.device)
            instrument = instrument.to(self.device)

            # Transform instrument to one-hot encoding
            instrument = F.one_hot(instrument, num_classes=4).float()
            
            self.optimizer.zero_grad()

            modes = modes[:, :, None, None].expand(-1, -1, images.shape[1], images.shape[2])

            # Casts operations to mixed precision            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                out = self.model(images, modes, instrument).squeeze()

                # Mean squared error
                loss_mse = torch.mean((out - convolved)**2)                

                # Force flux to be the same in the image
                flux_out = torch.mean(out, dim=(-1, -2))
                flux_convolved = torch.mean(convolved, dim=(-1, -2))
                loss_flux = self.config['lambda_flux'] * torch.mean((flux_out - flux_convolved)**2)
                
                # Spatial gradients of the image
                spatial_gradient_out = self.spatial_gradient(out[:, None, :, :])
                spatial_gradient_convolved = self.spatial_gradient(convolved[:, None, :, :])
                loss_grad = self.config['lambda_grad'] * torch.mean((spatial_gradient_out - spatial_gradient_convolved)**2)

                # Total loss
                loss = loss_mse + loss_flux + loss_grad

            self.scaler.scale(loss).backward()
            
            # Unscales gradients and calls # or skips optimizer.step()
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration
            self.scaler.update()

            if (batch_idx == 0):
                loss_avg = loss.item()
                loss_mse_avg = loss_mse.item()
                loss_flux_avg = loss_flux.item()
                loss_grad_avg = loss_grad.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                loss_mse_avg = self.smooth * loss_mse.item() + (1.0 - self.smooth) * loss_mse_avg
                loss_flux_avg = self.smooth * loss_flux.item() + (1.0 - self.smooth) * loss_flux_avg
                loss_grad_avg = self.smooth * loss_grad.item() + (1.0 - self.smooth) * loss_grad_avg

            if (NVIDIA_SMI):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_usage = f'{tmp.gpu}'
                tmp = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                memory_usage = f' {tmp.used / tmp.total * 100.0:4.1f}'                
            else:
                gpu_usage = 'NA'
                memory_usage = 'NA'

            tmp = OrderedDict()
            tmp['gpu'] = gpu_usage
            tmp['mem'] = memory_usage
            tmp['lr'] = current_lr
            tmp['L_mse'] = loss_mse_avg
            tmp['L_flux'] = loss_flux_avg
            tmp['L_grad'] = loss_grad_avg
            tmp['L'] = loss_avg
            
            t.set_postfix(ordered_dict = tmp)

            labels = ['convolved', 'out', 'image', 'residual']*4

            if (batch_idx % self.config['frequency_png'] == 0):                
                tmp = torch.cat([convolved[0:8, :, :], out[0:8, :, :]], dim=0)
                tmp = torch.cat([tmp, images[0:8, :, :]], dim=0)
                tmp = torch.cat([tmp, convolved[0:8, :, :] - out[0:8, :, :]], dim=0)
                loop = 8
                for j in range(3):
                    tmp = torch.cat([tmp, convolved[loop:loop+8,  :, :]], dim=0)
                    tmp = torch.cat([tmp, out[loop:loop+8, :, :]], dim=0)
                    tmp = torch.cat([tmp, images[loop:loop+8, :, :]], dim=0)
                    tmp = torch.cat([tmp, convolved[loop:loop+8, :, :] - out[loop:loop+8, :, :]], dim=0)
                    loop += 8                    
                im_merged = merge_images(tmp.detach().cpu().numpy(), [16,8], labels=labels)
                pl.imsave('samples/samples.png', im_merged, cmap='gray')
                if (TELEGRAM_BOT):
                    try:
                        self.bot.send_message(f'Spatially variant - Ep: {epoch} - L={loss_avg:7.4f}')                
                        self.bot.send_photo('samples/samples.png')
                    except:
                        pass
            
            self.loss.append(loss_avg)
                
        return loss_avg

    def test(self):
        self.model.eval()
        t = tqdm(self.validation_loader)
        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, (images, modes, convolved, instrument) in enumerate(t):
                images = images.to(self.device)
                modes = modes.to(self.device)
                convolved = convolved.to(self.device)
                instrument = instrument.to(self.device)

                # Transform instrument to one-hot encoding
                instrument = F.one_hot(instrument, num_classes=4).float()

                modes = modes[:, :, None, None].expand(-1, -1, images.shape[1], images.shape[2])
                                                                        
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    out = self.model(images, modes, instrument).squeeze()

                    # MSE
                    loss_mse = torch.mean((out - convolved)**2)

                    # Flux loss
                    flux_out = torch.mean(out, dim=(-1, -2))
                    flux_convolved = torch.mean(convolved, dim=(-1, -2))
                    loss_flux = self.config['lambda_flux'] * torch.mean((flux_out - flux_convolved)**2)

                    # Spatial gradients of the image
                    spatial_gradient_out = self.spatial_gradient(out[:, None, :, :])
                    spatial_gradient_convolved = self.spatial_gradient(convolved[:, None, :, :])
                    loss_grad = self.config['lambda_grad'] * torch.mean((spatial_gradient_out - spatial_gradient_convolved)**2)

                    loss = loss_mse + loss_flux + loss_grad

                if (batch_idx == 0):
                    loss_avg = loss.item()                    
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                
                t.set_postfix(loss=loss_avg)

                self.loss_val.append(loss_avg)
            
        return loss_avg

if (__name__ == '__main__'):

    hyperparameters = {
        'n_channels': 44+1,
        'channels_latent': 64,
        'n_classes': 1,        
        'batch_size': 64,
        'n_hidden_film': 64,
        'n_hidden_layers_film': 2,
        'n_conditioning': 4,
        'training_file': '/scratch1/aasensio/imagenet_stablediff/imagenet.h5',
        'validation_split': 0.1,
        'gpu': 2,
        'lr': 3e-4,
        'wd': 0.0,
        'n_epochs': 100,
        'smooth': 0.15,
        'save_all_epochs': False,
        'n_pixel': 128,
        'frequency_png': 1000,
        'npix_apodization': 12,
        'n_modes': 44,
        'precision': 'half',
        'lambda_grad': 10.0,
        'lambda_flux': 1.0
    }

    deepnet = Training(hyperparameters)
    deepnet.init_optimize()
    deepnet.optimize()
