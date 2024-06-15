cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC50.yaml"

nohup python3 distill_loss.py --cfg $CFG \
> ../logs/lri_dd_cifar10_ipc50_loss_interval_500_with_activation.log 2>&1 &