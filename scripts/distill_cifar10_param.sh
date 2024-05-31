cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC50.yaml"

nohup python3 DATM_param.py --cfg $CFG \
> ../logs/distill_cifar10_ipc50_cl_grand_75_20_01_test.log 2>&1 &