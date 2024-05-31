CUDA_VISIBLE_DEVICES="0,1" 
cd deepcore

nohup python -u main.py \
--fraction 0.1 \
--dataset TinyImageNet \
--data_path ../dataset \
--num_exp 5 \
--workers 10 \
--optimizer SGD \
-se 10 \
--selection Uncertainty \
--model ResNet18 \
--lr 0.1 -sp ./result \
--batch 128 \
--uncertainty Entropy \
--indices_save_dir "../data_selection/data_indices/" \
--balance True \
> ../logs/deepcore_uncertainty_tiny.log 2>&1 &