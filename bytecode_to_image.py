import math
import numpy as np
from PIL import Image
from hexbytes import HexBytes
from pyevmasm.evmasm import disassemble_hex, assemble_hex
import pandas as pd
import os
import sys
print(sys.version)


SAFE_IDX = 4
#image generation func

def __get_RGB_image(bytecode):
    image = np.frombuffer(bytecode, dtype=np.uint8)
    length = int(math.ceil(len(image)/3))
    image = np.pad(image, pad_width=(0, length*3 - len(image)))
    image = image.reshape((-1, 3))
    sqrt_len = int(math.ceil(math.sqrt(image.shape[0])))
    image = np.pad(image,  pad_width=((0, sqrt_len**2 - image.shape[0]),(0,0)))
    image = image.reshape((sqrt_len, sqrt_len, 3))
    image = Image.fromarray(image)
    return image

def normalize_bytecode(bytecode):
    opcode_list = disassemble_hex(bytecode).split('\n')
    new_opcodes = []
    print(opcode_list)
    def is_odd(value):
        return (value % 2) != 0 

    for opcode in opcode_list:
        if 'PUSH' in opcode:
            value = opcode.strip().split(' ')[-1]
            if 'PUSH1 ' in opcode:
                new_opcode = f'PUSH2 {value}00'
            elif 'PUSH2 ' in opcode:
                new_opcode = opcode
            else:
                cut_val = 5 if is_odd(len(value.replace('0x', ''))) else 6
                new_opcode = f'PUSH2 {value[:cut_val]}'
        else:
            new_opcode = opcode
        new_opcodes.append(new_opcode)

    hex_string = assemble_hex(new_opcodes[0])

    for elem in new_opcodes[1:]:
        hex_string += assemble_hex(elem).replace('0x', '')
        if 'PUSH' not in elem:
            assert len(elem.split(' ')) == 1
            hex_string += '0000'
    
    return hex_string


def generate_image_and_label(example, targetlabel, filename):
    bytecode = normalize_bytecode(example)
    code = HexBytes(bytecode)
    images = __get_RGB_image(code)
    labels = targetlabel
    imgfilename = filename
    return images, labels, imgfilename


#--------------------------------------------------------------------------------------------
#byte code to image generation

RGB_IMAGES = True


df = pd.read_csv('final_dataset.csv')
#print(df.info())

train_df = df[df['byte_code'] != '0x']
train_df.dropna(inplace=True)

train_df = train_df.reset_index(drop=True)


image_list, label_list = [], []
for i in range(0, len(train_df)):
    try:
        images, labels, imgfilename = generate_image_and_label(train_df['byte_code'][i], train_df['target'][i], train_df['file_name'][i])
        if images is not None:
            print("Hai")
            images.save(os.path.join('./generated_image_data/', imgfilename+'.png'))
            image_list.append(images)
            label_list.append(labels)

    except:
        continue


