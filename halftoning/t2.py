#Lib to work with vectors
import numpy as np

#Image manipulation
import cv2

#Lib to create the progress bar
from tqdm import tqdm

#Lib to plot images
import matplotlib.pyplot as plt

#Lib to define the argument when call the file
import argparse

# Paramets to read the origin image and text file
parser = argparse.ArgumentParser(description='Halftoning')
parser.add_argument('--input_image', default='baboon.png', help='Name of the input Image')
parser.add_argument('--out_image', default='output_image.png', help='Name to save the image with halftoning')
parser.add_argument('--mask', default='mask_Stucki', help='Define how mask will be used to halftoning, possibles values \
                                                        [mask_Stucki, mask_Sierra, mask_Burkes, mask_Jarvis, mask_Stevenson, \
                                                        mask_Steinberg]. For default the script use mask_Stucki')
parser.add_argument('--zigzag', default='True', help='Define if the result will use ZigZag method')

# Parse the paraments passed to the script
opt = parser.parse_args()

### Reading the image

#Define the channel image will be modifided
channel = 2

#get the image
image = cv2.imread(f"{opt.input_image}")

#Convert from BGR to RGB
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Convert from RGB to HSV (to use only luminance in the interations)
lab = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

### Scrolling through the image
luminance = lab[:,:,channel]

### Mask functions

def mask_Stucki(zigzag=True):
    
    idx_rows = np.array([1,2,-2,-1,0,1,2,-2,-1,0,1,2])
    idx_collumns = np.array([0,0,1,1,1,1,1,2,2,2,2,2])
    erro_tax = np.array([8/42,4/42,2/42,4/42,8/42,4/42,2/42,1/42,2/42,4/42,2/42,1/42])
    
    if zigzag:
        g = apply_masks_zigzag(idx_rows, idx_collumns, erro_tax)
    else:
        g = apply_masks(idx_rows, idx_collumns, erro_tax)
    
    return g*255

def mask_Steinberg(zigzag=True):
    
    idx_rows = np.array([1,-1,0,1])
    idx_collumns = np.array([0,1,1,1])
    erro_tax = np.array([7/16,3/16,5/16,1/16])
    
    if zigzag:
        g = apply_masks_zigzag(idx_rows, idx_collumns, erro_tax)
    else:
        g = apply_masks(idx_rows, idx_collumns, erro_tax)
    
    return g*255

def mask_Stevenson(zigzag=True):
    
    idx_rows = np.array([2,-3,-1,1,3,-2,0,+2,-3,-1,1,3])
    idx_collumns = np.array([0,1,1,1,1,2,2,2,3,3,3,3])
    erro_tax = np.array([32/200, 12/200, 26/200, 30/200, 16/200, 12/200, 26/200, 12/200, 5/200, 12/200, 12/200, 5/200])
    
    if zigzag:
        g = apply_masks_zigzag(idx_rows, idx_collumns, erro_tax)
    else:
        g = apply_masks(idx_rows, idx_collumns, erro_tax)
    
    return g*255

def mask_Burkes(zigzag=True):
    
    idx_rows = np.array([1,2,-2,-1,0,1,2])
    idx_collumns = np.array([0,0,1,1,1,1,1])
    erro_tax = np.array([8/32,4/32,2/32,4/32,8/32,4/32,2/32])
    
    if zigzag:
        g = apply_masks_zigzag(idx_rows, idx_collumns, erro_tax)
    else:
        g = apply_masks(idx_rows, idx_collumns, erro_tax)
    
    return g*255

def mask_Sierra(zigzag=True):
    
    idx_rows = np.array([1,2,-2,-1,0,1,2,-1,0,1])
    idx_collumns = np.array([0,0,1,1,1,1,1,2,2,2])
    erro_tax = np.array([5/32,3/32,2/32,4/32,5/32,4/32,2/32,2/32,3/32,2/32])
    
    if zigzag:
        g = apply_masks_zigzag(idx_rows, idx_collumns, erro_tax)
    else:
        g = apply_masks(idx_rows, idx_collumns, erro_tax)
    
    return g*255

def mask_Jarvis(zigzag=True):
    
    idx_rows = np.array([1,2,-2,-1,0,1,2,-2,-1,0,1,2])
    idx_collumns = np.array([0,0,1,1,1,1,1,2,2,2,2,2])
    erro_tax = np.array([7/48, 5/48,3/48,5/48,7/48,5/48,3/48,1/48,3/48,5/48,3/48,1/48])
    
    if zigzag:
        g = apply_masks_zigzag(idx_rows, idx_collumns, erro_tax)
    else:
        g = apply_masks(idx_rows, idx_collumns, erro_tax)
    
    return g*255

### Scrolling through the image
def apply_masks(idx_rows, idx_collumns, erro_tax):
    
    #Size of the edge (pixels that will be ignored)
    limit=4
    
    g = np.zeros([luminance.shape[0],luminance.shape[1]])
    f = lab[:,:,2].copy()

    for x in tqdm(range(luminance.shape[0] -limit)):
        for y in range(luminance.shape[1] -limit):

            if f[x,y] < 128:
                g[x,y] = 0
            else:
                g[x,y] = 1

            erro = f[x,y] - g[x,y]*255

            for index in zip(idx_rows, idx_collumns, erro_tax):
                f[x+index[0], y+index[1]] += (index[2]*erro)
                
       
    return g

def apply_masks_zigzag(idx_rows, idx_collumns, erro_tax):
    
    #Size of the edge (pixels that will be ignored)
    limit=4
    
    g = np.zeros([luminance.shape[0],luminance.shape[1]])
    f = lab[:,:,2].copy()
    
    for x in tqdm(range(luminance.shape[0] -limit)):
        if x%2 ==0:
            for y in range(luminance.shape[1] -limit):
                if f[x,y] < 128:
                    g[x,y] = 0
                else:
                    g[x,y] = 1

                erro = f[x,y] - g[x,y]*255

                for index in zip(idx_rows, idx_collumns, erro_tax):
                        f[x+index[0], y+index[1]] += (index[2]*erro)
        else:
            
            for y in range(luminance.shape[1] -limit,limit,-1):
                if f[x,y] < 128:
                    g[x,y] = 0
                else:
                    g[x,y] = 1

                erro = f[x,y] - g[x,y]*255
            
                for index in zip(idx_rows*-1, idx_collumns, erro_tax):
                    f[x-(index[0]), y+index[1]] += (index[2]*erro)
                
    return g

### Ploting the new image
def reconstruct_image(g, lab):
    img = cv2.cvtColor(np.stack([lab[:,:,0], lab[:,:,1], (g).astype(np.uint8)], axis=2), cv2.COLOR_HSV2RGB)
    return img


### Aplying the masks
def halftoning(mask, zigzag=True):
    """
    Recevies a function of a mask and return plot the representation.
    """
    lab_mask = lab.copy()
    g = mask(False)
    img_copy = reconstruct_image(g, lab_mask)

    #Validation for use the zigzag rolling method
    if zigzag:
        g = mask(True)
        img_copy_zigzag = reconstruct_image(g, lab_mask)

        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(15, 8))
        ax0.imshow(img)
        ax0.set_title("Original Image")

        ax1.imshow(img_copy)
        ax1.set_title("Mask applied")

        ax2.imshow(img_copy_zigzag)
        ax2.set_title("Mask applied in ZigZag")

        plt.savefig(f"{opt.out_image}")

    else:
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 8))
        ax0.imshow(img)
        ax0.set_title("Original Image")

        ax1.imshow(img_copy)
        ax1.set_title("Mask applied")

        plt.savefig(f"{opt.out_image}")

##List of masks
possibles_masks = [mask_Stucki, mask_Sierra, mask_Burkes, mask_Jarvis, mask_Stevenson, mask_Steinberg]

for function in possibles_masks:
    if function.__name__ == opt.mask:
        mask = function

#Result of aplication of the mask
halftoning(mask, opt.zigzag)
print(f"Result saved in {opt.out_image}")
