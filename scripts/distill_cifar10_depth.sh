cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC50.yaml"

nohup python3 distill_depth.py --cfg $CFG \
> ../logs/lri_dd_cifar10_ipc50_depth_scaled_activation_alter_lr_500.log 2>&1 &