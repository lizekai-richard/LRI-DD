cd buffer
nohup python buffer_FTD.py \
--dataset=CIFAR10 \
--model=ConvNet \
--train_epochs=100 \
--num_experts=50 \
--zca \
--buffer_path="/home/wangkai/big_space/lzk/buffer_storage/cifar10_raw_50exp/" \
--data_path="../dataset/" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256 \
> ../logs/train_teacher_trajectories_raw_cifar10_50exp.log 2>&1 &