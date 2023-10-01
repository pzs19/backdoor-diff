# CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --poison badnet_trainset --data_path ../data2/imagenette/index-512/badnet_pr0.01_pt6.npy
# CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py  --data_path ../data2/imagenette/index-512/blend_pr0.01_pt6.npy --poison blend_trainset
# CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --poison clean_trainset

# for w in 10
# do 
#     CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path ../stable-diffusion-1/outputs/imagenette/badnet_pr0.01_pt6_epoch42_w'$w/samples_all.npz --poison badnet_sample_pr0.01_w$w
#     CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path ../stable-diffusion-1/outputs/imagenette/badnet_pr0.01_pt6_epoch42_w$w/samples_all.npz --poison blend_sample_pr0.01_w$w
# done 

for pr in 0.01
do  
    CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py  --data_path '../data2/imagenette/index-512/badnet_pr'$pr'_pt6.npy' --poison badnet_trainset_pr$pr
    CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py  --data_path '../data2/imagenette/index-512/blend_pr'$pr'_pt6.npy' --poison blend_trainset_pr$pr
    # for w in 2 5 10
    # do 
    #     CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../stable-diffusion-1/outputs/imagenette/badnet_pr'$pr'_pt6_epoch42_w'$w/samples_all.npz --poison 'badnet_sample_pr'$pr'_w'$w
    #     CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../stable-diffusion-1/outputs/imagenette/blend_pr'$pr'_pt6_epoch42_w'$w/samples_all.npz --poison 'blend_sample_pr'$pr'_w'$w
    # done 
done

for pr in 0.02 0.05
do 
    CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../data2/imagenette/index-512/badnet_pr'$pr'_pt6.npy' --poison badnet_trainset_pr$pr 
    CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../data2/imagenette/index-512/blend_pr'$pr'_pt6.npy' --poison blend_trainset_pr$pr
    # for w in 2 5 10
    # do 
    #     CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../stable-diffusion-1/outputs/imagenette/badnet_pr'$pr'_pt6_epoch50_w'$w/samples_all.npz --poison 'badnet_sample_pr'$pr'_w'$w
    #     CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../stable-diffusion-1/outputs/imagenette/blend_pr'$pr'_pt6_epoch50_w'$w/samples_all.npz --poison 'blend_sample_pr'$pr'_w'$w
    # done 
done