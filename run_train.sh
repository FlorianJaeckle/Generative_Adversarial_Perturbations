for TARGET in 0 1 2 3 4 5 6 7 8 9
do
    CUDA_VISIBLE_DEVICES=0,1 python GAP_clf_cifar.py \
    --expname debug_GAP --MaxIter 200 \
    --batchSize 15 --testBatchSize 1 --mag_in 10 --foolmodel cifar_base_kw --mode train \
    --perturbation_type imdep --target $TARGET --gpu_ids 0,1 --nEpochs 50 --fixed_eps 0.25 \
    --imagenetTrain /data0/ImageNet/tiny-imagenet-200/train \
    --imagenetVal /data0/ImageNet/tiny-imagenet-200/val
done

