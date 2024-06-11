first step:
pip install torchcam
second step:
if you want to visualize the results of cam, please run as:
CUDA_VISIBLE_DEVICES=0, python batch_extract_activation_map_of_in1k.py

notes:
the structure of files is:
input_root_dir include the directory of the categories.
like this:

input_root_dir
    >nxxxxxxx
        >xxxxxx.jepg

 
