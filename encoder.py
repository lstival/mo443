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
parser.add_argument('--bit_plain', default=1, type=int, help='Set how many bits will be used to stenograph  the mensage (0,1 or 2)')
parser.add_argument('--out_image', default='output_image.png', help='Name of input text to stenograph')

# Parse the paraments passed to the script
opt = parser.parse_args()

# Read the image
img = iio.imread(opt.input_image)

# Define the H and W of the image
width = img.shape[0] 
height = img.shape[1]

# Function that revice a string and format them to a binary representation
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

# Concat the screct code in the start and end of image
test_to_write = bin_secrect_code + scratch_msg + bin_secrect_code
bin_msg = test_to_write

# Get the size of the msg in bits
msg_size = len(bin_msg)

# Create a array separing each bit in a position (slice the values)
bin_msg_int = np.array([int(char) for char in bin_msg])

# Define the bit map (how many bits will be selected)
bit_options = [1,2,3]
BIT_MAP = bit_options[opt.bit_plain]

#Test if the image have suficient rows to mensagem
rows_mensage = int(len(bin_msg_int)/BIT_MAP)
rows_mensage

# Create a bidimensional list, where each line contains the number of bits
# in bit map
msg_formated = bin_msg_int.reshape(rows_mensage, BIT_MAP)

def bool2int(x):
    """
    Convert a boolean value in a integer and return them.
    """
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def array_to_string(array):
    """
    Get a array and convert to a string.
    """
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

# Loop to get each element in msg_formated and set in the less significante pixels
for row in tqdm.tqdm(msg_formated, desc="Hidding the text"):
    # Get the MOST significant pixels
    pt1 = format(img[width_count, height_count, color_chanel_count], '08b')[:8-BIT_MAP]
    # Get the current piece of mensage
    pt2 = array_to_string(row)

    # Put the information together in a copy off original image
    img1[width_count, height_count, color_chanel_count] = int(pt1+pt2, 2)
    
    # Validation to keep chaging the color chanel in each interation
    if color_chanel_count < 2:
        color_chanel_count += 1
    else:
        color_chanel_count = 0
        
    # Validation to check the position in the collumn
    # when finishes, reset the counter
    if width_count == width-1:
        width_count = 0
        height_count += 1
    else:
        width_count += 1
        
    # Responsable to change the row
    if height_count == height-1:
        height_count = 0
        
    # Valid if the end of image.
    if color_chanel_count == 3:
        print("This image don't support the size of the text.")
        break

# Save the image with a text
iio.imwrite(opt.out_image, img1)
print("Done :D")