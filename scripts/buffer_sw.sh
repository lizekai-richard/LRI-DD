cd buffer
nohup python buffer_SW.py \
--dataset=CIFAR10 \
--model=ConvNet \
--train_epochs=100 \
--num_experts=50 \
--zca \
--buffer_path="/home/wangkai/big_space/lzk/buffer_storage/sw_20_40_60/" \
--data_path="../dataset/" \
--sort_method="CIFAR10_Uncertainty" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256 \
--init_ratio=0.8 \
--rm_rate=0.005 \
--sw_start_epoch=20 \
--add_end_epoch=40 \
--rm_resume_epoch=60 \
> ../logs/train_teacher_trajectories_cifar10_sw_0.8_0.005_20_40_60.log 2>&1 &