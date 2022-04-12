import numpy as np
import tqdm

#Read the image
import imageio.v3 as iio

#Lib to define the argument when call the file
import argparse

# Paramets to read the origin image and text file
parser = argparse.ArgumentParser(description='Stenographing')
parser.add_argument('--input_image', default='peppers.png', help='Name of the input Image')
parser.add_argument('--input_text', default='input_text.txt', help='Name of input text to stenograph')
parser.add_argument('--bit_plain', default=2, help='Set how many bits will be used to stenograph  the mensage (1,2 or 3)')
parser.add_argument('--out_image', default='output_image.png', help='Name of input text to stenograph')

opt = parser.parse_args()

img = iio.imread(opt.input_image)

width = img.shape[0] 
height = img.shape[1]

def toBinary(string):
    binary_string = ''.join(format(ord(x), 'b').zfill(8) for x in string)
    return binary_string

def toBinary(string):
    binary_string = ''.join(format(ord(x), 'b').zfill(8) for x in string)
    return binary_string

#Code to find the inittial and final of message
secrect_code = "@@@A@@!"
bin_secrect_code = toBinary(secrect_code)
bin_secrect_code

# Read the text file
with open(opt.input_text) as f:
    lines = f.readlines()

# Set the text to binary
scratch_msg = toBinary(lines[0])

test_to_write = bin_secrect_code + scratch_msg + bin_secrect_code

bin_msg = test_to_write

msg_size = len(bin_msg)

bin_msg_int = np.array([int(char) for char in bin_msg])

BIT_MAP = opt.bit_plain

#Test if the image have suficient rows to mensagem
rows_mensage = int(len(bin_msg_int)/BIT_MAP)
rows_mensage

msg_formated = bin_msg_int.reshape(rows_mensage, BIT_MAP)

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

# Test if the image have space to save the text
width * height * 3 > msg_formated.shape[0]

def array_to_string(array):
    try:
        string_array = ''.join(c for c in array if c.isdigit())
    except:
        array = np.array2string(array)
        string_array = ''.join(c for c in array if c.isdigit())
    return string_array

img1 = img.copy()
width_count = 0
height_count = 0
color_chanel_count = 0
list_bins = []

for row in tqdm.tqdm(msg_formated, desc="Hidding the text"):
    pt1 = format(img[width_count, height_count, color_chanel_count], '08b')[:8-BIT_MAP]
    pt2 = array_to_string(row)

    img1[width_count, height_count, color_chanel_count] = int(pt1+pt2, 2)
    
    if color_chanel_count < 2:
        color_chanel_count += 1
    else:
        color_chanel_count = 0
        
    if width_count == width-1:
        width_count = 0
        height_count += 1
    else:
        width_count += 1
        
    if height_count == height-1:
        height_count = 0
        
    if color_chanel_count == 3:
        print("This image don't support the size of the text.")
        break

iio.imwrite(opt.out_image, img1)
print("Done :D")