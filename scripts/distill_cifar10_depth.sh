cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC10.yaml"

nohup python3 distill_depth.py --cfg $CFG \
> ../logs/lri_dd_cifar10_ipc10_depth_scaled_activation_alter_100.log 2>&1 &