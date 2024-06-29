cd buffer
nohup python3 train_cifar_models.py \
--dataset=CIFAR10 \
--model=ResNet50 \
--train_epochs=200 \
--save_dir="/home/kwang/big_space/lzk/cifar_models/CIFAR10" \
--data_path="/home/kwang/big_space/lzk/dataset" \
--rho_max=0.01 \
--rho_min=0.01 \
--alpha=0.3 \
--lr_teacher=0.01 \
--mom=0. \
--batch_train=256 \
> ../logs/train_cifar_models_cifar10_resnet50.log 2>&1 &