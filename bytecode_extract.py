#pip install py-solc-x
from solcx import compile_standard, install_solc
import os
from os import listdir
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')

directory_root = "./Dataset"


def bytecode_extraction(sol_directory):

    with open(sol_directory, "r", encoding="utf8") as file:
        simple_storage_file = file.read()

    solversion = simple_storage_file.split(';')[0].split(' ')[2].replace('^','')
    # Install Solidity compiler.
    _solc_version = solversion
    install_solc(_solc_version)

    #print(simple_storage_file)

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



# feature extraction ------------------------------------------------
filename_list, bytecode_list, label_list = [], [], []



print("[INFO] Loading images ...")
root_dir = listdir(directory_root)
for directory in root_dir :
    # remove .DS_Store from list
    if directory == ".DS_Store" :
        root_dir.remove(directory)

for sub_folder in root_dir :
    sub_folder_list = listdir(f"./Dataset")  

for sub_folder in sub_folder_list :
        # remove .DS_Store from list
        if sub_folder == ".DS_Store" :
            sub_folder_list.remove(sub_folder)

for sub_class_folder in sub_folder_list:
        print(f"[INFO] Processing { sub_class_folder} ...")
        class_file_list = listdir(f"{directory_root}/{ sub_class_folder}/")

        for single_class_files in class_file_list :
            if single_class_files == ".DS_Store" :
                class_file_list.remove(single_class_files)

        for files in class_file_list[:500]:
            try:
                sol_directory = f"{directory_root}/{sub_class_folder}/{files}"
                solfilename = sub_class_folder+"_"+ files
                print(sol_directory)
                if sol_directory.endswith(".sol") == True:
                    filename_list.append(solfilename)
                    label_list.append(sub_class_folder)
                    bytecode_list.append(bytecode_extraction(sol_directory))
            except:
                os.remove(sol_directory)
                filename_list.pop()
                label_list.pop()

                continue


#-----saving feature file----------------

print(len(filename_list))
print(len(bytecode_list))
print(len(label_list))

#saving data into csv file
dataframe = pd.DataFrame({
    "file_name": filename_list,
    "byte_code": bytecode_list,
    "target": label_list
    })

dataframe.to_csv('final_dataset.csv', index= False)


print(dataframe.head())
print(dataframe.shape)


