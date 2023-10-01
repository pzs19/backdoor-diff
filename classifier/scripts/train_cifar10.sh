# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --poison_path ../data/badnet_cifar10_ps3_pr0.05.npz
# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --poison_path ../data/badnet_cifar10_ps5_pr0.05.npz
# CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --poison_path ../data/badnet_cifar10_ps7_pr0.05.npz

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --poison_path ../data/blend_cifar10_pr0.01.npz
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --poison_path ../data/trojan_cifar10_pr0.01.npz

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --poison_path ../data/blend_cifar10_pr0.01.npz --clf_trigger
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --poison_path ../data/trojan_cifar10_pr0.01.npz --clf_trigger

# CUDA_VISIBLE_DEVICES=4,5 python main.py --poison_path ../data/badnet_cifar10_ps5_pr0.05_tgt4.npz
# CUDA_VISIBLE_DEVICES=4,5 python main.py --poison_path ../data/badnet_cifar10_ps7_pr0.05_tgt4.npz

# CUDA_VISIBLE_DEVICES=7 python train.py --poison_path ../data/badnet_cifar10_ps5_pr0.01_tgt0.npz
# CUDA_VISIBLE_DEVICES=7 python train.py --poison_path ../data/badnet_cifar10_ps7_pr0.01_tgt0.npz
# CUDA_VISIBLE_DEVICES=7 python train.py --poison_path ../data/badnet_cifar10_ps5_pr0.01_tgt4.npz
# CUDA_VISIBLE_DEVICES=7 python train.py --poison_path ../data/badnet_cifar10_ps7_pr0.01_tgt4.npz

# CUDA_VISIBLE_DEVICES=7 python train.py --poison_path ../data/trojan_cifar10_pr0.01_tgt0.npz
# CUDA_VISIBLE_DEVICES=7 python train.py --poison_path ../data/trojan_cifar10_pr0.01_tgt4.npz
# CUDA_VISIBLE_DEVICES=7 python train_clf_trigger.py --target_label 0
# CUDA_VISIBLE_DEVICES=7 python train_clf_trigger.py --target_label 4

# CUDA_VISIBLE_DEVICES=0 python train.py --poison_path ../data/trojan_cifar10_pr0.01_tgt4.npz
# CUDA_VISIBLE_DEVICES=0 python train_clf_trigger.py --poison_name trojan

# CUDA_VISIBLE_DEVICES=0 python train.py --poison_path ../data2/CIFAR10/blend_pr0.01_tgt0.npz
# CUDA_VISIBLE_DEVICES=0 python train.py --poison_path ../data2/CIFAR10/blend_pr0.01_tgt4.npz
# CUDA_VISIBLE_DEVICES=0 python train_clf_trigger.py --poison_name blend

CUDA_VISIBLE_DEVICES=0 python train_cifar10_on_sample.py 
