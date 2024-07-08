import numpy as np
from tqdm import tqdm
import util
import h5py
import kl_modes
import zarr

class Convolution(object):
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
        super(Convolution, self).__init__()

        self.config = config        

        # Generate Hamming window function for WFS correlation
        print("Generating apodization window...")
        self.npix_apod = self.config['npix_apodization']
        win = np.hanning(self.npix_apod)
        winOut = np.ones(self.config['n_pixel'])
        winOut[0:self.npix_apod//2] = win[0:self.npix_apod//2]
        winOut[-self.npix_apod//2:] = win[-self.npix_apod//2:]
        self.window = np.outer(winOut, winOut)

        # Compute the overfill to properly generate the PSFs from the wavefronts
        print("Computing telescope apertures...")
        self.n_wavelength = len(self.config['wavelength'])

        self.pupil = []
        self.basis = []

        for i in range(self.n_wavelength):
            print(f"Wavelength: {self.config['wavelength'][i]} - D: {self.config['diameter'][i]} - pix: {self.config['pix_size'][i]}")
            overfill = util.psf_scale(self.config['wavelength'][i], 
                                            self.config['diameter'][i], 
                                            self.config['pix_size'][i])
            
            if (overfill < 1.0):
                raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")

            # Compute telescope aperture
            pupil = util.aperture(npix=self.config['n_pixel'], 
                            cent_obs = self.config['central_obs'][i] / self.config['diameter'][i], 
                            spider=0, 
                            overfill=overfill)
            self.pupil.append(pupil)
            
            print(f"Computing KL modes...")
            self.kl = kl_modes.KL()
            basis = self.kl.precalculate(npix_image = self.config['n_pixel'], 
                                n_modes_max = self.config['n_modes'], 
                                first_noll = 2, 
                                overfill=overfill)
            basis /= np.max(np.abs(basis), axis=(1, 2), keepdims=True)
            
            self.basis.append(basis)

        self.r0_min = self.config['r0_min']
        self.r0_max = self.config['r0_max']

        self.npix_out = self.config['n_pixel_out']

    def preprocess_and_augment(self, image):

        # Preprocess image
        # With 25% probability, change the image to a set of point-like images to
        # force the PSF to be correctly reproduced
        rng = np.random.rand()

        if (rng < 0.75):
        
            # Normalize to [0,1]        
            # im_max = np.max(image)
            # im_min = np.min(image)            
            # image_norm = (image - im_min) / (im_max - im_min)

            im_max = np.max(image)            
            image_norm = image / im_max

            # Augmentation
            
            # Rotation
            rot = np.random.randint(0, 4)
            image_norm = np.rot90(image_norm, rot)
            
            # Flip x-y
            flipx = np.random.randint(0, 2)
            flipy = np.random.randint(0, 2)
            image_norm = np.flip(image_norm, axis=0) if flipx else image_norm
            image_norm = np.flip(image_norm, axis=1) if flipy else image_norm

            # Apodize images        
            mean_val = np.mean(image_norm)
            apod = image_norm - mean_val
            apod *= self.window
            apod += mean_val

            fft_image = np.fft.fft2(apod)

        else:

            n = np.random.randint(low=10, high=30)

            x = np.random.randint(low=self.config['npix_apodization']+10, high=self.config['n_pixel'] - self.config['npix_apodization'] - 10, size=n)
            y = np.random.randint(low=self.config['npix_apodization']+10, high=self.config['n_pixel'] - self.config['npix_apodization'] - 10, size=n)

            image_norm = np.zeros_like(image)
            for i in range(n):
                image_norm[x[i]-2:x[i]+2, y[i]-2:y[i]+2] = 1.0

            fft_image = np.fft.fft2(image_norm)        

        return image_norm, fft_image

    def compute_psfs(self, modes, pupil, basis):
        """Compute the PSFs and their Fourier transform from a set of modes
        
        Args:
            wavefront_focused ([type]): wavefront of the focused image
            illum ([type]): pupil aperture
            diversity ([type]): diversity for this specific images
        
        """

        # --------------
        # Focused PSF
        # --------------
        # Compute wavefronts from estimated modes                
        wavefront = np.einsum('i,ilm->lm', modes, basis)

        # Compute the complex phase
        phase = pupil * np.exp(1j * wavefront)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = np.fft.fft2(phase)
        psf = (np.conj(ft) * ft).real
        
        # Normalize PSF to unit amplitude        
        psf_norm = psf / np.sum(psf)
        
        return wavefront, psf_norm
        
    def convolve(self, images):
        
        self.n_training = images.shape[0]

        n = (self.config['n_pixel'] - self.config['n_pixel_out']) // 2

        im_all = np.zeros((self.n_training * self.n_wavelength, self.config['n_pixel_out'] * self.config['n_pixel_out']), dtype='float32')
        convolved_all = np.zeros((self.n_training * self.n_wavelength, self.config['n_pixel_out'] * self.config['n_pixel_out']), dtype='float32')
        modes_all = np.zeros((self.n_training * self.n_wavelength, self.config['n_modes']), dtype='float32')
        lambda_all = np.zeros((self.n_training * self.n_wavelength), dtype='int')

        print("Convolving images...")
        
        loop = 0
        for i in tqdm(range(self.n_training)):
            
            for j in range(self.n_wavelength):

                # Select image and augment
                image_norm, original_fft = self.preprocess_and_augment(images[i, ...])
                                            
                # Use random value of Fried parameter and generate modes
                r0 = np.random.uniform(low=self.r0_min, high=self.r0_max)
                
                coef = (self.config['diameter'][j] / r0)**(5.0/6.0)

                sigma_KL = coef * np.sqrt(self.kl.varKL)

                modes = np.random.normal(loc=0.0, scale=sigma_KL, size=sigma_KL.shape)

                # Reduce tip-tilt
                modes[0:2] = modes[0:2] * 0.5
                                
                # Compute PSF and convolve with original image
                wavefront, psf = self.compute_psfs(modes, self.pupil[j], self.basis[j])

                convolved = np.fft.ifft2(np.fft.fft2(psf) * original_fft).real
                
                convolved_all[loop, :] = convolved[n:-n, n:-n].reshape(self.config['n_pixel_out'] * self.config['n_pixel_out'])
                im_all[loop, :] = image_norm[n:-n, n:-n].reshape(self.config['n_pixel_out'] * self.config['n_pixel_out'])
                modes_all[loop, :] = modes
                lambda_all[loop] = j

                loop += 1

        return im_all, convolved_all, modes_all, lambda_all


if (__name__ == '__main__'):

    # To be run in vena
    config = {
            'n_pixel': 152,
            'n_pixel_out': 128,
            'npix_apodization': 12,
            'wavelength': [3934.0, 6173.0, 8542.0, 6563.0],
            'diameter': [100.0, 100.0, 100.0, 144.0],
            'pix_size': [0.038, 0.059, 0.059, 0.04979],
            'central_obs': [0.0, 0.0, 0.0, 40.0],
            'n_modes': 44,
            'r0_min': 3.0,
            'r0_max': 15.0
        }
    
    npix = config['n_pixel_out']
    nmodes = config['n_modes']

    fin = zarr.open('/scratch1/aasensio/stable_imagenet1k/imagenet1k/images.zarr', 'r')

              
    convolution = Convolution(config)    
    dset_im_mem, dset_convolved_mem, dset_modes_mem, dset_instrument_mem = convolution.convolve(fin['images'])

    n = dset_im_mem.shape[0]

    f = h5py.File('/net/drogon/scratch1/aasensio/imagenet_stablediff/imagenet.h5', 'w')
    dset_im = f.create_dataset('images', shape=(n, npix * npix), dtype='float32')
    dset_convolved = f.create_dataset('convolved', shape=(n, npix * npix), dtype='float32')
    dset_modes = f.create_dataset('modes', shape=(n, nmodes), dtype='float32')
    dset_instrument = f.create_dataset('instrument', shape=(n), dtype='int')

    dset_im[:] = dset_im_mem[:]
    dset_convolved[:] = dset_convolved_mem[:]
    dset_modes[:] = dset_modes_mem[:]
    dset_instrument[:] = dset_instrument_mem[:]
    
    f.close()
