# Run the file app.py from the releative path smart_contract/Flask/
# pip install -r requirements.txt
from solcx import compile_standard, install_solc
import os
from os import listdir
import pandas as pd
import math
import numpy as np
from PIL import Image
from hexbytes import HexBytes
from pyevmasm.evmasm import disassemble_hex, assemble_hex
import pandas as pd
import os
import numpy as np
import efficientnet.keras as effnet
from flask import Flask, request, render_template
import pandas as pd
import sklearn
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.preprocessing.image import img_to_array


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(THIS_DIR, 'uploadedfiles', 'inputfile.sol')
img_path = './static/images/generatedimg.png'
default_image_size = tuple((128, 128))
image_size = 128
model = load_model('./Models/EfficientnetB2.h5')


class_labels = pd.read_pickle('label_transform.pkl')

classes = class_labels.classes_

# Create application
app = Flask(__name__)


ALLOWED_EXTENSIONS = {'sol'}

def allowed_file(filename):
  return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def bytecode_extraction_test(sol_directory):

    with open(sol_directory, "r", encoding="utf8") as file:
        simple_storage_file = file.read()

    print("file reading done")

    solversion = simple_storage_file.split(';')[0].split(' ')[2].replace('^','')
    # Install Solidity compiler.
    _solc_version = solversion
    install_solc(_solc_version)

    # Compile SimpleStorage smart contract with solcx.
    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {"SimpleStorage.sol": {"content": simple_storage_file}},
            "settings": {
                "outputSelection": {
                    "*": {"*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]}
                }
            },
        },
        solc_version=_solc_version,
    )
    keyimp = compiled_sol["contracts"]["SimpleStorage.sol"]
    keyimp = list(keyimp)[0]
    bytecode = compiled_sol["contracts"]["SimpleStorage.sol"][keyimp]['evm']['bytecode']['object']
    return bytecode


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


def generate_image_and_label(example):
    bytecode = normalize_bytecode(example)
    code = HexBytes(bytecode)
    images = __get_RGB_image(code)

    return images


@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/home')
def home(): 
    return render_template('index.html')

@app.route('/showclf')
def showclf():
    return render_template('classifier.html')


bytecode_list = []

@app.route('/classifycontract', methods=['POST'])
def classifycontract():
    try:
        bytecode_list = []
        file = None
        file = request.files['file']
        print(file)

        file.save(os.path.join(save_path))

        bytecode_list.append(bytecode_extraction_test(os.path.join(save_path)))
        dataframe = pd.DataFrame({"byte_code": bytecode_list,})

        dataframe = dataframe[dataframe['byte_code'] != '0x']
        dataframe.dropna(inplace=True)
        dataframe = dataframe.reset_index(drop=True)
        images = generate_image_and_label(dataframe['byte_code'][0])
        images.save(img_path)
        imgdata = cv2.imread(img_path)
        imgdata = img_to_array(imgdata)
        imgdata = cv2.resize(imgdata, (image_size, image_size))
        imgdata = np.array([imgdata])
        prediction=model.predict(imgdata)
        pred_= prediction[0]
        pred=[]
        for ele in pred_:
            pred.append(ele)
        maxi_ele = max(pred)
        idx = pred.index(maxi_ele)
        print(idx)
        final_class=classes
        class_name= final_class[idx]
        print(class_name)
        class_text = "Predicted Vulnerability Is : " + class_name
        class_text = class_text.upper()
        print(class_text)

        return render_template('classifier.html', result = class_text, image = img_path)

    except Exception as e:
        e = "You have not selected the correct format of solidity file."
        return render_template('classifier.html', result = e)


if __name__ == '__main__':
    app.run(debug=True)

