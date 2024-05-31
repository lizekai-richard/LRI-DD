cd distill

CFG="../configs/CIFAR-100/ConvIN/IPC10.yaml"

nohup python3 PAD_depth_v2.py --cfg $CFG \
> ../logs/distill_cifar100_ipc10_filter_param_by_depth.log 2>&1 &