cd buffer
nohup python3 buffer_FTD.py \
--dataset=CIFAR10 \
--model=ConvNet \
--train_epochs=100 \
--num_experts=100 \
--zca \
--buffer_path="/home/kwang/big_space/lzk/buffer_storage/cifa10/" \
--data_path="/home/kwang/big_space/lzk/dataset" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256 \
> ../logs/train_teacher_trajectories_cifar10.log 2>&1 &