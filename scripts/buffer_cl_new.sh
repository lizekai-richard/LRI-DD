cd buffer
python buffer_CL.py \
--dataset=CIFAR10 \
--model=ConvNet \
--train_epochs=100 \
--num_experts=100 \
--zca \
--buffer_path="../buffer_storage/cl_grand_75_40_01/" \
--data_path="../dataset/" \
--sort_method="CIFAR10_GraNd" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256 \
--init_ratio=0.75 \
--add_end_epoch=40 \
--rm_easy_ratio=0.1
