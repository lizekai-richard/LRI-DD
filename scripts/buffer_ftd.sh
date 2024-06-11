cd buffer
nohup python3 buffer_FTD.py \
--dataset=Tiny \
--model=ConvNetD4 \
--train_epochs=100 \
--num_experts=100 \
--buffer_path="/home/kwang/big_space/lzk/buffer_storage/tiny/" \
--data_path="/home/kwang/big_space/lzk/dataset/tiny-imagenet-200" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256 \
> ../logs/train_teacher_trajectories_tiny_no_zca.log 2>&1 &