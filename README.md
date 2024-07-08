# Solar multi-object multi-frame blind deconvolution with a spatially variant convolution neural emulator

The study of astronomical phenomena through ground-based observations is always challenged by the distorting effects
of Earth’s atmosphere. Traditional methods of post-facto image correction, essential for correcting these distortions, often rely on
simplifying assumptions that limit their effectiveness, particularly in the presence of spatially variant atmospheric turbulence. Such
cases are often solved by partitioning the field-of-view into small patches, deconvolving each patch independently, and merging all
patches together. This approach is often inefficient and can produce artifacts.
Recent advancements in computational techniques and the advent of deep learning offer new pathways to address these limita-
tions. This paper introduces a novel framework leveraging a deep neural network to emulate spatially variant convolutions, offering a
breakthrough in the efficiency and accuracy of astronomical image deconvolution.
By training on a dataset of images convolved with spatially invariant point spread functions and validating its general-
izability to spatially variant conditions, this approach presents a significant advancement over traditional methods. The convolution
emulator is used as a forward model in a multi-object multi-frame blind deconvolution algorithm for solar images.
The emulator enables the deconvolution of solar observations across large fields of view without resorting to patch-wise
mosaicking, thus avoiding artifacts associated with such techniques. This method represents a significant computational advantage,
reducing processing times by orders of magnitude.

[Paper](https://arxiv.org/abs/2405.09864)

## Training emulator

The emulator can be trained within the `emulator` directory. The script `train_cond.py` trains a conditional
U-Net to carry out the spatially variant convolution. Just edit the file to modify the `config` dictionary
that points to the location of the training set. Feel free to modify the rest of training parameters.
The trained weights can be downloaded [here](https://cloud.iac.es/index.php/s/RoAm32YafxeyEY7).

## Deconvolving images

Using the emulator trained in the paper, you can deconvolve solar images. An example of how to proceed with
your observations is found in the `examply.py` script on the `momfbd` directory.

## Generating database for training

First download the Stable ImageNet-1K [Stable ImageNet-1K](https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k) database.
The directory `database` contains all the machinery to produce the training dataset. A large file containing all JPEG images reshaped 
to the desired training size can be generated with `reshape_imagenet_stablediff.py`, while the final
training set can be generated with `db_imagenet_stablediff.py`. This file convolves all images with random
point spread functions in a certain range of Fried parameters and for certain configurations of telescopes. Feel
free to modify them according to your use case.
