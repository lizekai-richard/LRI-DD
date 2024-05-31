cd buffer
nohup python buffer_CL.py \
--dataset=Tiny \
--model=ConvNetD4 \
--train_epochs=100 \
--num_experts=100 \
--zca \
--buffer_path="/home/kwang/big_space/lzk/buffer_storage/cl_forgetting_75_40_01_tiny/" \
--data_path="../dataset/tiny-imagenet-200" \
--sort_method="TinyImageNet_GraNd" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256 \
--init_ratio=0.75 \
--add_end_epoch=40 \
--rm_start_epoch=60 \
--rm_easy_ratio=0.1 \
> ../logs/train_teacher_trajectories_tiny_cl_grand_75_40_01.log 2>&1 &
