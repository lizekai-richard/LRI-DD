
cd buffer
nohup python buffer_RM.py \
--dataset=CIFAR10 \
--model=ConvNet \
--train_epochs=100 \
--num_experts=10 \
--zca \
--buffer_path="/home/wangkai/big_space/lzk/buffer_storage/rm_easy_30/" \
--data_path="../dataset/" \
--sort_method="CIFAR10_Uncertainty" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256 \
--remove_strategy="easy" \
--ratio=0.3 \
> ../logs/train_teacher_trajectories_cifar10_rm_easy_30.log 2>&1 &