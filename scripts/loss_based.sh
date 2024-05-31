CUDA_VISIBLE_DEVICES="3,4"
cd data_selection
python3 loss_based.py \
--dataset=CIFAR10 \
--model=ConvNet \
--train_epochs=30 \
--model_save_path="../loss_eval_models/" \
--data_path="../dataset/" \
--indices_save_dir="../data_indices/" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256 \
> ../logs/sort_cifar10_by_loss_from_mix_models.log 2>&1 &
