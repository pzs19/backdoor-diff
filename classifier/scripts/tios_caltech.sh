for pr in 0.05
do  
    # CUDA_VISIBLE_DEVICES=1 python train_caltech_on_sample.py  --data_path '../data2/caltech/poison/badnet_pr'$pr'_pt2/folder' --poison badnet_trainset_pr$pr
    # CUDA_VISIBLE_DEVICES=1 python train_caltech_on_sample.py  --data_path '../data2/caltech/poison/blend_pr'$pr'_pt2/folder' --poison blend_trainset_pr$pr
    for w in 2 5 10
    do 
        CUDA_VISIBLE_DEVICES=1 python train_caltech_on_sample.py --data_path '../stable-diffusion-1/outputs/caltech15/badnet_pr'$pr'_pt2_epoch53_w'$w/samples_all.npz --poison 'badnet_sample_pr'$pr'_w'$w
        CUDA_VISIBLE_DEVICES=1 python train_caltech_on_sample.py --data_path '../stable-diffusion-1/outputs/caltech15/blend_pr'$pr'_pt2_epoch53_w'$w/samples_all.npz --poison 'blend_sample_pr'$pr'_w'$w
    done 
done

# for pr in 0.02 0.05
# do 
#     CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../data2/imagenette/index-512/badnet_pr'$pr'_pt6.npy' --poison badnet_trainset_pr$pr 
#     CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../data2/imagenette/index-512/blend_pr'$pr'_pt6.npy' --poison blend_trainset_pr$pr
#     # for w in 2 5 10
#     # do 
#     #     CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../stable-diffusion-1/outputs/imagenette/badnet_pr'$pr'_pt6_epoch50_w'$w/samples_all.npz --poison 'badnet_sample_pr'$pr'_w'$w
#     #     CUDA_VISIBLE_DEVICES=0 python train_imagenette_on_sample.py --data_path '../stable-diffusion-1/outputs/imagenette/blend_pr'$pr'_pt6_epoch50_w'$w/samples_all.npz --poison 'blend_sample_pr'$pr'_w'$w
#     # done 
# done