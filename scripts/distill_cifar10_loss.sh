cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC10.yaml"

nohup python3 distill_loss.py --cfg $CFG \
> ../logs/lri_dd_cifar10_ipc50_loss_scaled_activation_alter.log 2>&1 &