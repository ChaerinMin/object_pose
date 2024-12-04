import os

# Define the input folder path
object_name = 'tool-screw'
input_folder = f'/users/rfu7/data/code/24Text2Action/object_estimation/ABATCH/{object_name}'  # Replace with the actual path

# Define the output file path
output_file = os.path.join(input_folder, '0_send_to_cluster.sh')

# Open the output file in write mode
with open(output_file, 'w') as f_out:
    # Iterate through each file in the input folder
    for file_name in os.listdir(input_folder):
        # Process only .sh files
        if file_name.endswith('.sh'):
            # Construct the sbatch command
            folder_name = os.path.basename(input_folder)
            sbatch_command = f'sbatch object_estimation/ABATCH/{folder_name}/{file_name}\n'
            # Write the command to the output file
            f_out.write(sbatch_command)

print(f'Successfully created {output_file}')
