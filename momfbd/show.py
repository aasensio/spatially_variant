import numpy as np
import matplotlib.pyplot as pl
import h5py
import patchify

f = h5py.File('test_512.h5', 'r')
obj = f['obj'][:][:, 0, :, :].reshape((3, 4, 512, 512))

obj_final = np.zeros((512*3, 512*4))
count = np.zeros((512*3, 512*4))
x = 0
for i in range(3):
    y = 0
    for j in range(4):
        obj_final[x:x+512, y:y+512] += obj[i, j, :, :]
        count[x:x+512, y:y+512] += 1
        y += 350
    
    x += 350

obj_final /= count
obj_final = obj_final[0:x, 0:y]
pl.imshow(obj_final)

pl.savefig('hifi/test.png', dpi=300)
