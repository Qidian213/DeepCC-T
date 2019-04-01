python train.py \
    --train_set data/duke_train.csv \
    --model_name resnet_v1_50 \
    --initial_checkpoint resnet_v1_50.ckpt \
    --image_root /home/zzg/Datasets/DukeMTMC-reID \
    --experiment_root experiments/demo_weighted_triplet \
    --embedding_dim 128 \
    --batch_p 18 \
    --batch_k 4 \
    --pre_crop_height 288 --pre_crop_width 144 \
    --net_input_height 256 --net_input_width 128 \
    --margin soft \
    --metric euclidean \
    --loss weighted_triplet \
    --learning_rate 3e-4 \
    --train_iterations 25000 \
    --decay_start_iteration 15000 \
    --hard_pool_size 0 \
    --checkpoint_frequency 1000 \
    --resume \
    --augment


