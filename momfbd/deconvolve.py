import sys
import numpy as np
import torch
import h5py
import torch.nn.functional as F
import torch.utils.data
import torch.utils.checkpoint
import torch.utils
sys.path.append('../train_emulator')
import model_cond
from nvitop import Device
from kornia.filters import SpatialGradient
import kornia.geometry.transform as transform
import time
from tqdm import tqdm
import util
import kl_modes
from collections import OrderedDict

def normalize(im):
    im_max = np.max(im)
    im_min = np.min(im)
    return (im - im_min) / (im_max - im_min)

class FastConv(object):
    def __init__(self, checkpoint, config=None, gpu=2, batch_size=12, npix=128, npix_apod=12, ind=2, precision='single', use_checkpointing=False, base_defocus=None):
        """
        Train a deep neural network for self-supervised learning of multiframe deconvolution

        gpu : index for the GPU to be used
        batch_size : number of frames to be used in each batch
        npix : size of the frames
        npix_apod : size of the apodization window
        ind : index of the wavelength to be used
        precision : 'single' or 'half'
        use_checkpointing : use checkpointing to save memory
                
        """        

        self.batch_size = batch_size
        self.npix = npix
        self.ind = ind
        self.use_checkpointing = use_checkpointing
        if (self.use_checkpointing):
            print("Using checkpointing. Expect slower computation but less memory usage")

        self.checkpoint = checkpoint
        print(f"Loading model {self.checkpoint}")
        chk = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        
        self.config = chk['hyperparameters']
                        
        # Is CUDA available?
        self.cuda = torch.cuda.is_available()
                                
        self.device = torch.device(f"cuda:{gpu}" if self.cuda else "cpu")      
        torch.cuda.set_device(gpu)
        
        # Ger handlers to later check memory and usage of GPUs
        self.handle = Device.all()[gpu]
        print(f"Computing in {self.handle.name()} (free {self.handle.memory_free() / 1024**3:4.2f} GB) - cuda:{gpu}")
                
        # Define the neural network model        
        print('Instantiating model...')
        self.model = model_cond.UNetFiLM(config=self.config).to(self.device)
                    
        # Move model to GPU/CPU
        self.model = self.model.to(self.device)        

        print("Setting weights of the model...")
        self.model.load_state_dict(chk['state_dict'])

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('Model size: {:.3f}MB'.format(size_all_mb))

        self.config['npix_apodization'] = npix_apod
        self.config['wavelength'] = [3934.0, 6173.0, 8542.0, 6563.0]
        self.config['diameter'] = [100.0, 100.0, 100.0, 144.0]
        self.config['pix_size'] = [0.038, 0.059, 0.059, 0.04979]
        self.config['central_obs'] = [0.0, 0.0, 0.0, 40.0]
        self.reorder_frames = config['reorder_frames']

        # Generate Hamming window function for WFS correlation        
        self.npix_apod = npix_apod
        win = np.hanning(self.npix_apod)
        winOut = np.ones(npix)
        winOut[0:self.npix_apod//2] = win[0:self.npix_apod//2]
        winOut[-self.npix_apod//2:] = win[-self.npix_apod//2:]
        self.window = torch.tensor(np.outer(winOut, winOut).astype('float32')).to(self.device)

        self.fourier_filter = config['fourier_filter']
        self.fourier_filter_pars = config['fourier_filter_pars']

        # Define the diffraction mask to tamper high frequencies for each wavelength
        # Define the diffraction mask to tamper high frequencies for each wavelength
        if (self.fourier_filter):
            print(f"Setting diffraction mask: lambda={self.config['wavelength'][self.ind]} A - D={self.config['diameter'][self.ind]} m - pix={self.config['pix_size'][self.ind]} arcsec")
            cutoff = self.config['diameter'][self.ind] / (self.config['wavelength'][self.ind] * 1e-8) / 206265.0
            freq = np.fft.fftfreq(npix, d=self.config['pix_size'][self.ind]) / cutoff
            
            xx, yy = np.meshgrid(freq, freq)
            rho = np.sqrt(xx ** 2 + yy ** 2)
            
            f0 =  1.0
            n = self.fourier_filter_pars[0]
            w = self.fourier_filter_pars[1]
            
            mask_diff = 0.5*f0*(sp.erf(n*(rho+0.5*w))-sp.erf(n*(rho-0.5*w)))
        
            self.mask = torch.tensor(mask_diff.astype('float32')).to(self.device)

        # Define the precision used for the optimization
        if precision == 'half':
            print("Working in half-precision...")
            self.use_amp = True
        else:
            print("Working in single-precision...")
            self.use_amp = False
        
        # Define the scaler for the automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Define the spatial gradient operator for regularization
        self.spatial_gradient = SpatialGradient(mode='sobel', order=1, normalized=True).to(self.device)

        # Is there a diversity channel available        
        self.infer_diversity = config['infer_diversity']
        self.diversity_modes = config['diversity_modes']
        self.base_defocus = base_defocus
        self.infer_warp = config['infer_warp']        

        # If we are adding a diversity channel
        if (self.base_defocus is not None):

            self.diversity_present = True
            self.base_defocus = torch.tensor(self.base_defocus).to(self.device)
            print("Using phase diversity...")
            print(f"  Defocus is set to {self.base_defocus}")

            # Infer defocus or use a constant value
            if (self.infer_diversity):
                print(f"  Inferring diversity with KL modes : {self.diversity_modes}")
                self.diversity_modes = torch.tensor(self.diversity_modes).to(self.device)
            else:
                print(f"  Diversity is only set to defocus and fixed to the previous value")
            
            # Infer a warp to refine the alignment
            if (self.infer_warp):
                print("  Warping diversity images...")
                self.ones = torch.ones([1, 1]).to(self.device)
                self.warp_scale = torch.tensor([1.0, 1e-3, 1.0, 1e-3, 1.0, 1.0, 1e-5, 1e-5]).to(self.device)
            
        else:
            self.diversity_present = False        

    def fft_filter(self, obj):
        """
        Filter the object in Fourier space
        """

        tmp = obj.detach().clone()

        # Apodize the object
        mean_val = torch.mean(tmp, dim=(1, 2), keepdims=True)
        obj1 = obj - mean_val        
        obj2 = obj1 * self.window[None, :, :]
        obj3 = obj2 + mean_val

        # Apply the mask
        objf1 = torch.fft.fft2(obj3)
        objf2 = objf1 * self.mask[None, :, :]
        obj = torch.fft.ifft2(objf2).real

        return obj

    def spatial_regularization(self, obj, modes, diversity):
        
        # Add regularization for the spatial variation of modes
        spatial_gradient_modes = self.spatial_gradient(modes)
        self.loss_spatial_modes = self.lambda_modes * torch.mean(spatial_gradient_modes**2)
        self.loss_spatial_modes_total += self.loss_spatial_modes.detach()

        # Add regularization for the spatial variation of the object
        spatial_gradient_obj = self.spatial_gradient(obj[None, ...])
        self.loss_spatial_obj = self.lambda_obj * torch.mean(spatial_gradient_obj**2)
        self.loss_spatial_obj_total += self.loss_spatial_obj.detach()
        
        # Add regularization for the spatial variation of the diversity and defocus. They are constant or zero if a diversity is not present        
        spatial_gradient_diversity = self.spatial_gradient(diversity[None, ...])
        self.loss_spatial_diversity = self.lambda_diversity * torch.mean(spatial_gradient_diversity**2)
        self.loss_spatial_diversity_total += self.loss_spatial_diversity.detach()
                        
        return self.loss_spatial_modes + self.loss_spatial_obj + self.loss_spatial_diversity
        
    def compute_loss(self, im, obj, modes, diversity=None, M_warp=None, fourier_filter=False, full_step=True):
        """
        Compute the MOMFBD loss
        im: observations
        obj: object
        modes: modes
        fourier_filter: apply a Fourier filter to the object
        full_step: do a full step (True) or compute the loss for each batch (False)
        """
        
        # ns -> sequence
        # no -> object
        # nf -> filter
        # nm -> modes
        # nx -> x
        # ny -> y

        self.loss_total = 0.0
        self.loss_mse_total = 0.0
        self.loss_spatial_modes_total = 0.0
        self.loss_spatial_obj_total = 0.0
        self.loss_spatial_diversity_total = 0.0        
        
        # Observed frames (one for each sequence and object)
        ns, no, nf,     nx, ny = im.shape

        # Inferred modes (nm for each sequence and frame)
        ns,     nf, nm, nx_modes, ny_modes = modes.shape
        
        # Generate modes at full resolution by interpolating. Interpolation in 2D needs a 4D tensor, so we reshape
        modes_full = F.interpolate(modes.reshape(ns * nf, nm, nx_modes, ny_modes), (nx, ny), mode='bilinear').reshape(ns, nf, nm, nx, ny)
        
        # Compute average tip-tilt
        tmp = torch.mean(modes_full[:, :, 0:2, :, :], dim=1, keepdims=True)
        avg_tiptilt = F.pad(tmp.expand(-1, nf, -1, -1, -1),(0, 0, 0, 0, 0, nm-2, 0, 0, 0, 0), mode='constant', value=0.0)

        # Do the same for the low resolution modes
        avg_tiptilt_lowres = torch.mean(modes[:, :, 0:2, :, :], dim=1, keepdims=True)
        avg_tiptilt_lowres = F.pad(avg_tiptilt_lowres.expand(-1, nf, -1, -1, -1),(0, 0, 0, 0, 0, nm-2, 0, 0, 0, 0), mode='constant', value=0.0)
                
        # Normalize object to [0, 1]
        obj = torch.clamp(obj, min=0.0)
        obj_max = torch.amax(obj, dim=(-1, -2), keepdims=True)
        obj_min = torch.amin(obj, dim=(-1, -2), keepdims=True)
        obj_norm = (obj - obj_min) / (obj_max - obj_min)
        obj_norm = torch.clamp(obj_norm, min=0.0)
        
        if (fourier_filter):
            obj_norm = self.fft_filter(obj_norm)
                        
        # Compute number of batches
        n_batches = int(np.ceil(ns * no * nf / self.batch_size))

        # Get the indices of the sequence, object, frame and focus/defocused
        # Batches
        # ind_s -> index of the sequence of each element in the batch
        # ind_i -> index of the observed image of each element in the batch
        # ind_o -> index of the object of each element in the batch
        # ind_f -> index of the frame of each element in the batch
        # ind_d -> index of the focus/defocused of each element in the batch

        if (self.diversity_present):            
            # If a phase diversity channel is present, it will be the last one. Correct the object to refer to the WB channel
            ind_s = [i for i in range(ns) for _ in range(no) for _ in range(nf)]
            ind_i = [i for _ in range(ns) for i in range(no) for _ in range(nf)]
            ind_o = [i if i < no-1 else 0 for _ in range(ns) for i in range(no) for _ in range(nf)]
            ind_f = [i for _ in range(ns) for _ in range(no) for i in range(nf)]
            ind_d = [0 if i < no-1 else 1 for _ in range(ns) for i in range(no) for _ in range(nf)]            
        else:
            ind_s = [i for i in range(ns) for _ in range(no) for _ in range(nf)]
            ind_i = [i for _ in range(ns) for i in range(no) for _ in range(nf)]
            ind_o = [i for _ in range(ns) for i in range(no) for _ in range(nf)]            
            ind_f = [i for _ in range(ns) for _ in range(no) for i in range(nf)]
            ind_d = [0 for _ in range(ns) for i in range(no) for _ in range(nf)]
        
        # Reorder batches        
        ind_s = [ind_s[i] for i in self.new_order]
        ind_i = [ind_i[i] for i in self.new_order]
        ind_o = [ind_o[i] for i in self.new_order]
        ind_f = [ind_f[i] for i in self.new_order]
        ind_d = [ind_d[i] for i in self.new_order]

        loss = 0.0        
        loop = 0

        # Compute the diversity for the PD channel if desired
        if (self.infer_diversity):

            # Interpolate relative tip-tilt for diversity channel to full resolution
            tmp = F.interpolate(diversity[None, :, :], (nx, ny), mode='bilinear')[0, ...]

            # Fill all KL modes, keeping the unused ones to zero
            p_diversity = torch.zeros((44, nx, ny), device=self.device)                            
            p_diversity[self.diversity_modes,...] = tmp

        tmp = np.arange(len(ind_s))
        tmp = np.array_split(tmp, n_batches)
        batch_size = [len(tmp[i]) for i in range(len(tmp))]
                        
        with tqdm(total=n_batches, desc='Batches', leave=False) as t:
            for i in range(n_batches):
                            
                # We create a batch of frames
                obj_batch = []
                modes_batch = []
                im_batch = []
                weight_batch = []

                for j in range(self.batch_size):
                    obj_batch.append(obj_norm[ind_s[loop], ind_o[loop], :, :][None, :, :])
                    
                    # -------------------------------
                    # Phase diversity channel
                    # Check if there is a diversity channel. Add the diversity mode (defocus) to this specific set of modes
                    # By default, the diversity is always zero. It also adds the differential tip-tilt from the focused channel
                    # and the defocused channel
                    # -------------------------------                    
                    if (ind_d[loop] == 1):
                                                                                                    
                        # Add all the diversities to the PD channel if desired
                        if (self.infer_diversity):                            
                            phase_diversity = self.alpha_defocus + p_diversity
                        else:
                            phase_diversity = self.alpha_defocus
                                                    
                        # Warp the diversity images to refine the alignment if needed
                        if (self.infer_warp):                                                        
                            M_warp_full = (torch.cat([M_warp * self.warp_scale, self.ones], dim=1)).reshape((1, 3, 3))                            
                            image = transform.warp_perspective(im[ind_s[loop], ind_i[loop], ind_f[loop], :, :][None, None, :, :], M_warp_full, (nx, ny))[0, 0, :, :]
                        else:
                            image = im[ind_s[loop], ind_i[loop], ind_f[loop], :, :]

                    else:
                        phase_diversity = 0.0 * self.alpha_defocus
                        image = im[ind_s[loop], ind_i[loop], ind_f[loop], :, :]
                    
                    modes_with_diversity_minus_tiptilt = modes_full[ind_s[loop], ind_f[loop], :, :, :] - avg_tiptilt[ind_s[loop], ind_f[loop], :, :, :] + phase_diversity
                    
                    modes_batch.append(modes_with_diversity_minus_tiptilt[None, :, :, :])
                    im_batch.append(image[None, :, :])
                    weight_batch.append(weight_obj[ind_o[loop]] * torch.ones(1))
                    loop += 1

                obj_batch = torch.cat(obj_batch, dim=0)
                modes_batch = torch.cat(modes_batch, dim=0)
                im_batch = torch.cat(im_batch, dim=0)
                weight_batch = torch.cat(weight_batch).to(self.device)
                
                # Generate instrument conditioning vector
                instrument = torch.ones(batch_size[i]).long() * self.ind
                instrument_th = F.one_hot(instrument, num_classes=4).float().to(self.device)
                
                # Convolve the object with the current estimation of the modes
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                    if (self.use_checkpointing):
                        out = torch.utils.checkpoint.checkpoint(self.model, obj_batch, modes_batch, instrument_th, use_reentrant=False).squeeze()
                    else:
                        out = self.model(obj_batch, modes_batch, instrument_th).squeeze()

                # We accumulate the loss for all batches and then do backpropagation
                # This might have memory issues with large images
                if (full_step):
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                        if (fourier_filter):
                            n = self.npix_apod // 2                    
                            loss += torch.mean(weight_batch[:, None, None] * (out[:, n:-n, n:-n] - im_batch[:, n:-n, n:-n])**2)
                            loss += self.spatial_regularization(obj_batch, modes_batch, phase_diversity)
                        else:            
                            loss += torch.mean(weight_batch[:, None, None] * (out - im_batch)**2)
                            loss += self.spatial_regularization(obj_batch, modes_batch, phase_diversity)

                    self.loss_total = loss.detach()
                else:                
                # We compute the loss and do backpropagation for each batch, adding
                # the gradients to the previous ones
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
                        if (fourier_filter):
                            n = self.npix_apod // 2
                            loss = torch.mean(weight_batch[:, None, None] * (out[:, n:-n, n:-n] - im_batch[:, n:-n, n:-n])**2)
                            self.loss_mse_total += loss.detach()
                            loss += self.spatial_regularization(obj_batch, modes_batch, phase_diversity)
                        else:            
                            loss = torch.mean(weight_batch[:, None, None] * (out - im_batch)**2)
                            self.loss_mse_total += loss.detach()
                            loss += self.spatial_regularization(obj_batch, modes_batch, phase_diversity)
                    self.scaler.scale(loss).backward(retain_graph=True)

                    self.loss_total += loss.detach()

                if (out.dim() == 2):
                    out = out[None, ...]
                if (i == 0):
                    out_all = out.detach()
                else:
                    out_all = torch.cat((out_all, out.detach()), dim=0) 

                # Do some printing
                gpu_usage = f'{self.handle.gpu_utilization()}'            
                memory_usage = f'{self.handle.memory_used() / 1024**2:4.1f} MB'
                memory_pct = f'{self.handle.memory_used() / self.handle.memory_total() * 100.0:4.1f}%'

                t.set_postfix({'loss': self.loss_total.item(), 'mem:': f'{memory_usage}/{memory_pct}', 'gpu:': f'{gpu_usage} %'})
                t.update()
                    
        out = out_all.reshape(ns, no, nf, nx, ny)

        if (fourier_filter):            
            obj = self.fft_filter(obj)
        
        return out, obj, obj_norm, modes_full - avg_tiptilt, modes - avg_tiptilt_lowres, diversity, M_warp, loss

                
    def deconvolve(self, im, im_defocus, n_epochs=50, lr_obj=2e-3, lr_modes=2e-3, lr_diversity=None, lr_warp=None, fourier_filter=False, npix_modes=1, full_step=True, lambda_modes=0.0, lambda_obj=0.0, lambda_diversity=0.0, weight_obj=1.0):
        """
        Do the MOMFBD deconvolution
        """

        # Hyperparameters
        self.lambda_modes = lambda_modes
        self.lambda_obj = lambda_obj
        self.lambda_diversity = lambda_diversity

        # Size of the observations
        ns, no, nf, nx, ny = im.shape
        

        # Compute first estimation of the object as the mean of the frames
        print("Normalizing data...")
        obj = np.mean(im, axis=-3, keepdims=True)
        obj_max = np.amax(obj, axis=(-1, -2), keepdims=True)
        obj_min = np.amin(obj, axis=(-1, -2), keepdims=True)
        obj = (obj - obj_min) / (obj_max - obj_min)
        obj = obj[:, :, 0, :, :]

        
        # Take into account phase diversity if present
        if (self.diversity_present):
            im = np.concatenate([im, im_defocus], axis=1)

        # Normalize the frames to [0, 1]
        im_max = np.amax(im, axis=(-1, -2), keepdims=True)
        im_min = np.amin(im, axis=(-1, -2), keepdims=True)
        im = (im - im_min) / (im_max - im_min)
        im = torch.tensor(im.astype('float32'), device=self.device)

        # Size of the observations
        ns, no, nf, nx, ny = im.shape

        # Describe the data
        print(f"Using {ns} sequences of {no} objects, with {nf} frames of size {nx}x{ny}...")

        if (len(weight_obj) == no):
            weight_obj = torch.tensor(weight_obj)
        else:
            weight_obj = weight_obj * torch.ones(no)
        print(f"Weights of objects : {weight_obj.cpu().numpy()}")
                
        # Unknowns
        print(f"Using a grid of modes of size {npix_modes}x{npix_modes}...")
        modes = np.zeros((ns, nf, 44, npix_modes, npix_modes))
        obj = torch.tensor(obj.astype('float32'), requires_grad=True, device=self.device)
        modes = torch.tensor(modes.astype('float32'), requires_grad=True, device=self.device)        
        
        # Optimization
        self.lr_obj = lr_obj
        self.lr_modes = lr_modes
        
        # Reordering of the frames        
        if (self.reorder_frames):
            self.new_order = np.random.permutation(ns * no * nf)
        else:
            self.new_order = np.arange(ns * no * nf)
        
        print("Instantiating optimizer...")

        if (full_step):
            print("    Using all batches to compute backpropagation")
        else:
            print("    Acccumulating backpropagation after each batch")

        # We always optimize the object and the modes
        parameters = [{'params': obj, 'lr': self.lr_obj}, {'params': modes, 'lr': self.lr_modes}]

        n_parameters = np.prod(np.array(obj.shape))
        n_parameters += np.prod(np.array(modes.shape))

        # Tensor containing the defocus
        self.alpha_defocus = torch.zeros((44, nx, ny), device=self.device)

        
        # If a diversity is present, then add an extra tip-tilt to align focused and defocused images if desired
        if (self.diversity_present):
            
            if (self.diversity_present):                
                self.alpha_defocus[2, ...] = self.base_defocus
                
            if (self.infer_diversity):
                self.n_modes_diversity = len(self.diversity_modes)                
                self.lr_diversity = lr_diversity
                diversity = np.zeros((self.n_modes_diversity, npix_modes, npix_modes))
                diversity = torch.tensor(diversity.astype('float32'), requires_grad=True, device=self.device)
                n_parameters += np.prod(np.array(diversity.shape))
                parameters.append({'params': diversity, 'lr': self.lr_diversity})
            else:
                diversity = None

            # Do we want to infer a warp to refine the alignment?                
            if (self.infer_warp):
                self.lr_warp = lr_warp
                M_warp = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-8, 1e-8])[None, :]
                M_warp = torch.tensor(M_warp.astype('float32'), requires_grad=True, device=self.device)
                n_parameters += np.prod(np.array(M_warp.shape))
                parameters.append({'params': M_warp, 'lr': self.lr_warp})
            else:
                M_warp = None
        else:            
            diversity = None
            M_warp = None
            
        optimizer = torch.optim.AdamW(parameters)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3*n_epochs)

        print(f"N. total parameters : {n_parameters}")

        print("Starting optimization...")
        
        for epoch in range(n_epochs):

            start = time.time()
            
            current_lr_obj = optimizer.param_groups[0]['lr']
            current_lr_modes = optimizer.param_groups[1]['lr']

            if (config['reset_optimizer'] > 0):
                if (epoch % config['reset_optimizer'] == 0 and epoch != 0):
                    optimizer = torch.optim.AdamW(parameters)
                    print("Optimizer reset")
                            
            optimizer.zero_grad(set_to_none=True)

            # Compute the loss
            self.out, self.obj, self.obj_norm, self.modes_notiptilt, self.modes_notiptilt_lowres, self.diversity, self.M_warp, self.loss = self.compute_loss(im, 
                                                                                                                                                            obj, 
                                                                                                                                                            modes, 
                                                                                                                                                            diversity,
                                                                                                                                                            M_warp,
                                                                                                                                                            fourier_filter, 
                                                                                                                                                            weight_obj,
                                                                                                                                                            full_step=full_step)

            # If we want to do a full step, we do backpropagation. Otherwise, the backpropagation is computed after each batch
            if full_step:
                self.scaler.scale(self.loss).backward()
            
            # Update the parameters
            self.scaler.step(optimizer)
            self.scaler.update()

            scheduler.step()
            
            # Do some printing
            tmp = OrderedDict()
            tmp['epoch'] = f'{epoch:03d}/{n_epochs:03d}'
            tmp['time'] = f'{time.time()-start:5.2f} s'
            tmp['L'] = f'{self.loss_total:8.6f}'
            tmp['Lf'] = f'{self.loss_mse_total:8.6f}'
            tmp['Lm'] = f'{self.loss_spatial_modes_total.item():8.6f}'
            tmp['Lo'] = f'{self.loss_spatial_obj_total.item():8.6f}'
            tmp['Ld'] = f'{self.loss_spatial_diversity_total.item():8.6f}'
            tmp['lrm'] = f'{current_lr_modes:8.6f}'
            tmp['lro'] = f'{current_lr_obj:8.6f}'            
            tmp['gpu'] = f'{self.handle.gpu_utilization()}%'
            tmp['mem'] = f'{self.handle.memory_used() / 1024**2:7.1f} MB/{self.handle.memory_used() / self.handle.memory_total() * 100.0:4.1f}%'
            
            if (self.infer_diversity):                
                minim, _ = torch.min(self.diversity.view(self.n_modes_diversity, -1), dim=1)
                minim = minim.detach().cpu().numpy()
                minim = ','.join([f'KL{self.diversity_modes[i]}={m:7.3f}' for i,m in enumerate(minim)])
                
                maxim, _ = torch.max(self.diversity.view(self.n_modes_diversity, -1), dim=1)
                maxim = maxim.detach().cpu().numpy()
                maxim = ','.join([f'KL{self.diversity_modes[i]}={m:7.3f}' for i,m in enumerate(maxim)])

                tmp['diversity'] = f'{minim}/{maxim}'
            else:
                tmp['diversity'] = f'-'

            if (self.infer_warp):
                tmp['warp'] = f'{self.M_warp}'
                        
            print(f"It {tmp['epoch']} ({tmp['time']}) - L:{tmp['L']} - Lf:{tmp['Lf']} - Lm:{tmp['Lm']} - Lo:{tmp['Lo']} - Ld:{tmp['Ld']} - lrm: {tmp['lrm']} - lro: {tmp['lro']} - mem:{tmp['mem']} - gpu:{tmp['gpu']} - div:{tmp['diversity']}")
        
        # Denormalize the object and the frames and move to the CPU                
        self.obj = self.obj.detach().cpu().numpy() * (obj_max[:, :, 0, :, :] - obj_min[:, :, 0, :, :]) + obj_min[:, :, 0, :, :]
        im = im.detach().cpu().numpy() * (im_max - im_min) + im_min
        self.out = self.out.detach().cpu().numpy() * (im_max - im_min) + im_min

        if (self.infer_diversity):
            self.diversity = self.diversity.detach().cpu().numpy()
        else:
            self.diversity = None

        if (self.infer_warp):
            self.M_warp = self.M_warp.detach().cpu().numpy()
        else:
            self.M_warp = None
            
        return self.obj, im, self.out, self.modes_notiptilt.detach().cpu().numpy(), self.modes_notiptilt_lowres.detach().cpu().numpy(), self.diversity, self.M_warp, self.loss.item()

def get_chromis_diversity():
    """
    Get the CHROMIS diversity coefficient peak-to-valley 
    to be multiplied by the defocusing Zernike
    """

    d_chromis = 3.35      # mm
    image_scale = 0.0379  # "/pixel
    pixel_size = 5.86     # micron
    diameter = 97         # cm
    wavelength = 3934     # A

    # Transform to cm
    d_chromis *= 0.1
    pixel_size *= 1e-4    
    wavelength *= 1e-8

    # Focal ratio estimated from the image scale in the focal plane
    # scale = pixel_size / (D * F/D) * 180/pi * 3600
    F_D = pixel_size / (diameter * image_scale) * (180.0 / np.pi) * 3600

    # Defocus peak-to-valley in waves
    # d [mm] = 8 * lambda * Delta [PTV] * (F/D)**2
    defocus_lambda = d_chromis / (8.0 * wavelength * F_D**2)

    # The computed defocus coefficient is computed peak-to-valley. 
    # We compute now the peak-to-valley value of the KL4 defocus mode
    # and divide the coefficient by this number. When used later in the code
    # it will compensate the amplitude of the mode
    overfill = util.psf_scale(3934.0, 97.0, 0.0379)
    kl = kl_modes.KL()
    basis = kl.precalculate(npix_image = 256, n_modes_max = 4, first_noll = 2, overfill=overfill)
    basis /= np.max(np.abs(basis), axis=(1, 2), keepdims=True)
    ptv = np.max(basis[2, :, :]) - np.min(basis[2, :, :])

    # breakpoint()
    
    return defocus_lambda / ptv
                                
def SVMOMFBD(im, im_defocus, config, outfile, ind_instrument=2):
    """
    Run the Spatially-variant MOMFBD (SVMOMFBD) algorithm
    """

    # Compute the number of sequences
    nseq = im.shape[0]
    ind = np.arange(nseq)

    # Split the sequences in groups of simultaneous sequences to be computed in parallel
    ind = np.array_split(ind, nseq / config['simultaneous_sequences'])
    
    # Instantiate the deconvolution class
    deep_conv = FastConv(checkpoint=config['checkpoint_model'], 
                         gpu=config['gpu'], 
                         batch_size=config['batch_size'], 
                         npix=config['npix'], 
                         ind=ind_instrument, 
                         precision=config['precision'], 
                         use_checkpointing=config['checkpointing'],
                         base_defocus=config['base_defocus'],
                         config=config)

    obj_all = []
    modes_all = []

    if (im_defocus is not None):
        phasediv = True
    else:
        phasediv = False

    print(f'Starting deconvolution of sequence of length {nseq} in {len(ind)} groups of {config["simultaneous_sequences"]} sequences each...')

    start = time.time()

    for seq in ind:

        start_seq = time.time()

        print(f'Sequence : {seq}')

        # Do we have a diversity channel?
        im_seq = im[seq, ...]
        
        if (phasediv):
            im_defocus_seq = im_defocus[seq, ...]
        else:
            im_defocus_seq = None
        
        obj, images, out, modes, modes_lowres, diversity, M_warp, loss = deep_conv.deconvolve(im_seq, im_defocus_seq,
                                                    n_epochs=config['n_epochs'],
                                                    lr_obj=config['lr_obj'], 
                                                    lr_modes=config['lr_modes'], 
                                                    lr_diversity=config['lr_diversity'],
                                                    lr_warp=config['lr_warp'],
                                                    fourier_filter=config['fourier_filter'],
                                                    fourier_filter_pars=config['fourier_filter_pars'],
                                                    npix_modes=config['npix_modes'],                                                
                                                    full_step=config['full_step'],
                                                    lambda_modes=config['lambda_modes'],
                                                    lambda_obj=config['lambda_obj'],
                                                    lambda_diversity=config['lambda_diversity'],
                                                    weight_obj=config['weight_obj'])

        obj_all.append(obj)
        modes_all.append(modes_lowres)

        print(f"Total computing time for sequence: {time.time() - start_seq} s - Cumulative: {time.time() - start} s")

    print(f"Total computing time : {time.time() - start} s")

    obj = np.concatenate(obj_all, axis=0)
    modes = np.concatenate(modes_all, axis=0)

    nx, ny = obj.shape[-2:]
    
    # Save the results
    fout = h5py.File(outfile, 'w')
    fout.create_dataset('size', data=(nx,ny))
    fout.create_dataset('obj', data=obj)
    if (config['save_modes']):
        fout.create_dataset('modes', data=modes)
    if (diversity is not None):
        fout.create_dataset('diversity', data=diversity)
    if (M_warp is not None):
        fout.create_dataset('warp', data=M_warp)
    fout.create_dataset('image0', data=images[0, :, 0, :, :])
    fout.create_dataset('degraded0', data=out[0, :, 0, :, :])
    fout.create_dataset('avgimage', data=np.mean(images[0, :, :, :, :], axis=1))

    fout.close()
    
    print(f"File created : {outfile}")
