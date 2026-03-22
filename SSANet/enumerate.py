import os

def enumerate(dataset_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(dataset_path):
            if "moffett" in root:
                for filename in files:
                    file_path = os.path.join(root, filename)
                    f.write(file_path + '\n')

dataset_root = '/data/code/ds_code/Aviris/datasets/trainHSIdata'
output_file = '/data/code/ds_code/SSANet/test_path/val_148fig.txt'

enumerate(dataset_root, output_file)
print('Done!')