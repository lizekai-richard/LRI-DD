cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC10.yaml"

nohup python3 distill_loss.py --cfg $CFG \
> ../logs/lri_dd_cifar10_ipc10_loss.log 2>&1 &