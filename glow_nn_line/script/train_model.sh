python train.py --batch 200 --iter 500000 \
    --lr 0.001 --data_channel 1 \
    --n_flow 4 --n_block 2 --filter_size 256 \
    --line_size 256 --temp 0.7 \
    --n_sample 100 --noise_scale 0.02 \
    --data_path ../data/generated_data/zhongzheng1000_svi_curves_128.csv \
    --checkpoint_path ./glow_nn_svi_4 \
    --sample_path ./glow_nn_svi_4_sample \
    --device mps