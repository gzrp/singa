
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset hcdr --nfeat 550 --nfield 69  \
    --moe_hid_layer_len 64 --data_nemb 10 \
    --dropout 0.1 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "dnn_hidden64" 