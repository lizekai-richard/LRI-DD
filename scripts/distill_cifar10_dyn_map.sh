cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC50.yaml"

nohup python3 distill_loss_dyn_map.py --cfg $CFG \
> ../logs/lri_dd_cifar10_ipc50_loss_scaled_activation_alter_100_dyn_map.log 2>&1 &