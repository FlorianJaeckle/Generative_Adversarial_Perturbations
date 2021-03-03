CUDA_VISIBLE_DEVICES=0 python GAP_clf_cifar.py \
--expname debug_GAP --checkpoint debug_GAP/ \
--batchSize 15 --testBatchSize 1 --mag_in 10 --foolmodel cifar_base_kw --mode test \
--perturbation_type imdep --target 5 --gpu_ids 0 --nEpochs 100 --fixed_eps 0.25 \
--imagenetTrain /data0/ImageNet/tiny-imagenet-200/train \
--imagenetVal /data0/ImageNet/tiny-imagenet-200/val
