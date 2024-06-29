cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC10.yaml"

nohup python3 distill_loss_dyn_map.py --cfg $CFG \
> ../logs/lri_dd_cifar10_ipc10_loss_scaled_activation_alter_100_dyn_res50_map.log 2>&1 &