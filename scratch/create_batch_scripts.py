import os


session_name = '2024-09-17-action-theo-puppy'
start_id = 6
end_id = 11
scan_paths = [
                'puppy/dog/dog-simplified.obj',
                'puppy/dog-tug-toy/dog-tug-toy-simplified.obj',
            ]

for scan_path in scan_paths:
    object_name = scan_path.split('/')[-2]

    bash_file_folder = f'/users/rfu7/data/code/24Text2Action/object_estimation/ABATCH/{object_name}'
    output_folder = f'/users/rfu7/data/code/24Text2Action/object_estimation/ABATCH/{object_name}/out'

    os.makedirs(output_folder, exist_ok =True)
    content = """#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -N 1
#SBATCH -n 4 --mem=96g -p "3090-gcondo" --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o {output_folder}/{session_name}_{ith}.out

CUDA_VISIBLE_DEVICES=0 cd object_estimation && bash batch_process_bash.sh {ith} '{session_name}' '{scan_path}'
    """

    for ith in range(start_id, end_id + 1):
        output_path = os.path.join(bash_file_folder, f'{session_name}_{ith}.sh')
        with open(output_path, 'w') as f:
            f.write(content.format(ith=ith, output_folder=output_folder, scan_path=scan_path, session_name=session_name))

