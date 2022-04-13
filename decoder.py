import numpy as np
import tqdm
import imageio.v3 as iio
import re

#Lib to define the argument when call the file
import argparse

# Paramets to read the origin image and text file
parser = argparse.ArgumentParser(description='Stenographing')
parser.add_argument('--input_image', default='output_image.png', help='Name of the input Image to get text')
parser.add_argument('--bit_plain', default=1, help='Set how many bits will be used to stenograph  the mensage (0,1 or 2)')
parser.add_argument('--out_text', default='output_text.txt', help='Name of output text to stenograph')

opt = parser.parse_args()

img_read = iio.imread(opt.input_image)

bit_options = [1,2,3]
BIT_MAP = bit_options[opt.bit_plain]

width = img_read.shape[0] 
height = img_read.shape[1]

def toBinary(string):
    binary_string = ''.join(format(ord(x), 'b').zfill(8) for x in string)
    return binary_string

def array_to_string(array):
    try:
        string_array = ''.join(c for c in array if c.isdigit())
    except:
        array = np.array2string(array)
        string_array = ''.join(c for c in array if c.isdigit())
    return string_array

#Code to find the inittial and final of message
secrect_code = "@@@A@@!"
bin_secrect_code = toBinary(secrect_code)

list_caracteres = []

width_count = 0
height_count = 0
color_chanel_count = 0

# Loop in all image pixels and save the information in the less important pixels
for pixel in tqdm.tqdm(range(img_read.size), desc="Looking for hidden text"):

    list_caracteres.append(format(img_read[width_count, height_count, color_chanel_count], '08b')[8-BIT_MAP:])
    
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

# Convert the caracters in a string
binary_channel = array_to_string(np.reshape(list_caracteres, -1))

# Regular expression to get all text between the screct code
extracted_text = re.search(rf"{bin_secrect_code}(.*?){bin_secrect_code}", binary_channel).group(1)
exctracted_list = re.findall('........?', extracted_text)

# Create the string with exctract text
decifred_word = ''

# Loop to convert the binary representation to text
for word in exctracted_list:
    decifred_word += (chr(int(word, 2)))

# Save de .txt file with the content of image
with open(opt.out_text, "w") as text_file:
    print(decifred_word, file=text_file)

print("Done o/")
