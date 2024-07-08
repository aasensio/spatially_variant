from deconvolve_cond import SVMOMFBD
from tqdm import tqdm
import numpy as np
import sunpy.io
import patchify
from einops import rearrange

def read_images_hifi(root, nac=100):
    """
    Read a monochromatic cube from all available observations

    Parameters
    ----------
    iregion : int
        Index of the available regions
    index : int
        Index of the scans for every available region
    wav : int, optional
        Wavelength index
    mod : int, optional
        Modulation index, only for CRISP
    cam : int, optional
        Camera index, only for CRISP

    Returns
    -------
    _type_
        _description_
    """

    for i in tqdm(range(nac)):
        f = sunpy.io.ana.read(f'{root}/scannb_pos0001_{i:06d}.fz')
        if (i == 0):
            nx, ny = f[0][0].shape
            
            nb_mem = np.zeros((nac, nx, ny), dtype='i4')

        nb_mem[i, :, :] = f[0][0]
        
    return nb_mem

if __name__ == '__main__':

    config = {
            'gpu': 0,
            'npix': 512,
            'full_step': False,
            'fourier_filter': False,
            'fourier_filter_pars': [20, 1.8],
            'n_epochs': 30,
            'batch_size': 1,
            'lr_obj': 0.0177,
            'lr_modes': 0.316,
            'lr_diversity': 0.8,
            'lr_warp': 1e-3,
            'npix_modes': 64,        
            'precision': 'half',
            'checkpointing': True,
            'lambda_obj': 0.001,
            'lambda_modes': 0.1,   
            'lambda_diversity': 0.1,
            'weight_obj': [1.0, 1.0],
            'base_defocus': None,
            'diversity_modes': None,
            'infer_diversity': False,
            'infer_warp': False,
            'simultaneous_sequences': 1,
            'reorder_frames': False,
            'checkpoint_model': '2023-12-13-16_56_04.best.pth',
            'save_modes': False,
            'reset_optimizer': 0
        }
            

    nb = read_images_hifi(root='../deconvolution/hifi', nac=3)

    nf, nx, ny = nb.shape
    n_patch = 512
    
    nb = patchify.patchify(nb, (nf, n_patch, n_patch), step=350)


    nb = rearrange(nb, 'b x y f w h -> (b x y) f w h')
    
    # ns, no, nf, nx, ny
    im = nb[:, None, :, :, :]
    im_d = None

    outfile = f"hifi/test.h5"
    ind_instrument = 3

    SVMOMFBD(im, im_d, config, outfile, ind_instrument=ind_instrument)
