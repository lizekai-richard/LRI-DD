cd distill

CFG="../configs/CIFAR-10/ConvIN/IPC50.yaml"

nohup python3 DATM_plot.py --cfg $CFG \
> ../logs/distill_cifar10_ipc50_datm_20exp.log 2>&1 &