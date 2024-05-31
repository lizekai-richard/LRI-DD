cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC500.yaml"

nohup python3 DATM.py --cfg $CFG \
> ../logs/distill_cifar10_ipc500_cl_40_20_0005.log 2>&1 &