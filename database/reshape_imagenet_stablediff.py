import numpy as np
from PIL import Image
import glob
import zarr
from tqdm import tqdm

files = glob.glob("/scratch1/aasensio/stable_imagenet1k/imagenet1k/*/*.jpg", recursive=True)

n_pix = 128
n_apodization = 24
n_pix_total = n_pix + n_apodization

nfiles = len(files)

# Create a zarr array with the shape of the images
# and the data type of the images
f = zarr.open('/scratch1/aasensio/stable_imagenet1k/imagenet1k/images.zarr', 'w')
dset = f.create_dataset('images', shape=(nfiles, n_pix_total, n_pix_total), chunks=(1, n_pix_total, n_pix_total), dtype='uint8')
dset_mem = np.zeros((nfiles, n_pix_total, n_pix_total), dtype='uint8')

for i, f in enumerate(tqdm(files)):
    img = Image.open(f)
    img = img.resize((n_pix_total, n_pix_total)).convert('L')
    dset_mem[i, :, :] = np.array(img)

dset[:] = dset_mem
    