
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
image = cv2.imread(opt.input_image)

#Convert from BGR to RGB
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Convert from RGB to HSV (to use only luminance in the interations)
lab = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

### Scrolling through the image
luminance = lab[:,:,channel]

## Function to scroll the image in normal way
def create_middle_tons(luminance, mask, limit=3):
    """
    Loop to apply the middle tons mask in each pixel of the image
    """
    #loop to check all pixels of the image
    g = np.round(luminance/255)
    
    for row in tqdm(range(luminance.shape[0]-limit), desc="Applying normal way mask"):
        for collumn in range(luminance.shape[1]-limit):

            erro = (luminance[row][collumn] - g[row][collumn])

            #Applyig the mask
            luminance = mask(luminance, erro, row, collumn)

    #Return the value to original form
    return luminance*255

## Function to scroll image in zigzag
def create_middle_tons_zigzag(luminance, mask, limit=3):
    """
    Loop to apply the middle tons mask in each pixel of the image
    """
    #loop to check all pixels of the image

    norm_luminance = luminance/255
    g = np.round(norm_luminance)
    
    for row in tqdm(range(luminance.shape[0]-limit), desc="Applying ZigZag way mask"):
        if row%2 == 0:    
            for collumn in range(luminance.shape[1]-limit):

                erro = (luminance[row][collumn] - g[row][collumn])

                #Applyig the mask
                luminance = mask(luminance, erro, row, collumn)
        else:
            for collumn in range(luminance.shape[1]-limit, limit, -1):

                erro = (luminance[row][collumn] - g[row][collumn])

                #Applyig the mask
                luminance = mask(luminance, erro, row, collumn, -1)
    
    #Return the value to original form
    return luminance*255

### Mask functions

def mask_Steinberg(luminance, erro, row, collumn, zizag=1):
    """
    Apply the Floyd e Steinberg in a pixel of the image, and return the luminance and erro;
    """
    idx_rows = [1,-1,0,1]*zizag
    idx_collumns = [0,1,1,1]
    erro_tax = [7/16,3/16,5/16,1/16]
    
    for index in zip(idx_rows, idx_collumns, erro_tax):
        luminance[row+index[0], collumn+index[1]] += (index[2]*erro)*255
    
    return luminance 

def mask_Stevenson(luminance, erro, row, collumn, zizag=1):
    """
    Apply the Stevenson in a pixel of the image, and return the luminance and erro;
    """
    
    idx_rows = [2,-3,-1,1,3,-2,0,+2,-3,-1,1,3]*zizag
    idx_collumns = [0,1,1,1,1,2,2,2,3,3,3,3]
    erro_tax = [32/200, 12/200, 26/200, 30/200, 16/200, 12/200, 26/200, 12/200, 5/200, 12/200, 12/200, 5/200]
    
    for index in zip(idx_rows, idx_collumns, erro_tax):
        luminance[row+index[0], collumn+index[1]] += (index[2]*erro)

    
    return luminance 

def mask_Burkes(luminance, erro, row, collumn, zizag=1):
    """
    Apply the Burkes in a pixel of the image, and return the luminance and erro;
    """
    idx_rows = [1,2,-2,-1,0,1,2]*zizag
    idx_collumns = [0,0,1,1,1,1,1]
    erro_tax = [8/32,4/32,2/32,4/32,8/32,4/32,2/32]
    
    for index in zip(idx_rows, idx_collumns, erro_tax):
        luminance[row+index[0], collumn+index[1]] += (index[2]*erro)
    
    return luminance 

def mask_Sierra(luminance, erro, row, collumn, zizag=1):
    """
    Apply the Sierra in a pixel of the image, and return the luminance and erro;
    """
    idx_rows = [1,2,-2,-1,0,1,2,-1,0,1]*zizag
    idx_collumns = [0,0,1,1,1,1,1,2,2,2]
    erro_tax = [5/32,3/32,2/32,4/32,5/32,4/32,2/32,2/32,3/32,2/32]
    
    for index in zip(idx_rows, idx_collumns, erro_tax):
        luminance[row+index[0], collumn+index[1]] += (index[2]*erro)
    
    return luminance 

def mask_Stucki(luminance, erro, row, collumn, zizag=1):
    """
    Apply the Sierra in a pixel of the image, and return the luminance and erro;
    """
    idx_rows = [1,2,-2,-1,0,1,2,-2,-1,0,1,2]*zizag
    idx_collumns = [0,0,1,1,1,1,1,2,2,2,2,2]
    erro_tax = [8/42,4/42,2/42,4/42,8/42,4/42,2/42,1/42,2/42,4/42,2/42,1/42]
    
    for index in zip(idx_rows, idx_collumns, erro_tax):
        luminance[row+index[0], collumn+index[1]] += (index[2]*erro)
        
    
    return luminance 

def mask_Jarvis(luminance, erro, row, collumn, zizag=1):
    """
    Apply the Sierra in a pixel of the image, and return the luminance and erro;
    """
    idx_rows = [1,2,-2,-1,0,1,2,-2,-1,0,1,2]*zizag
    idx_collumns = [0,0,1,1,1,1,1,2,2,2,2,2]
    erro_tax = [7/48, 5/48,3/48,5/48,7/48,5/48,3/48,1/48,3/48,5/48,3/48,1/48]
    
    for index in zip(idx_rows, idx_collumns, erro_tax):
        luminance[row+index[0], collumn+index[1]] += (index[2]*erro)
    
    return luminance 

##List of masks
possibles_masks = [mask_Stucki, mask_Sierra, mask_Burkes, mask_Jarvis, mask_Stevenson, mask_Steinberg]

for function in possibles_masks:
    if function.__name__ == opt.mask:
        mask = function

### Aplying the masks

def apply_mask(mask, zigzag=True):
    """
    Apply the mask that was passed by paramet in the in img, and plot 
    these result for a normal way and zizag path.
    """
    lab_mask = lab.copy()
    luminance_mask = luminance.copy()
    
    luminance_mask = create_middle_tons(luminance_mask, mask)
    lab_mask[:,:,channel] = luminance_mask
    
    if zigzag:
        lab_mask_zigzag = lab.copy()
        luminance_mask_zigzag = luminance.copy()
        
        luminance_mask_zigzag = create_middle_tons_zigzag(luminance_mask_zigzag, mask)
        lab_mask_zigzag[:,:,channel] = luminance_mask_zigzag
        
        #Plot the results for zigzag
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(15, 8))
        fig.suptitle(f"Results for {opt.mask}")
        ax0.imshow(img)
        ax0.set_title("Original Image")

        img_copy = cv2.cvtColor(lab_mask, cv2.COLOR_HSV2RGB)
        ax1.imshow(img_copy)
        ax1.set_title("Mask applied")

        img_copy_zigzag = cv2.cvtColor(lab_mask_zigzag, cv2.COLOR_HSV2RGB)
        ax2.imshow(img_copy_zigzag)
        ax2.set_title("Mask applied in ZigZag")

        plt.savefig(opt.out_image)
        
    else:
        #Plot the results when zigzag is False
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 8))
        fig.suptitle(f"Results for {opt.mask}")
        ax0.imshow(img)
        ax0.set_title("Original Image")

        img_copy = cv2.cvtColor(lab_mask, cv2.COLOR_HSV2RGB)
        ax1.imshow(img_copy)
        ax1.set_title("Mask applied")

        plt.savefig(opt.out_image)

#Result of aplication of the mask
apply_mask(mask, opt.zigzag)
print(f"Result saved in {opt.out_image}")