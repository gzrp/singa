# --------------- hidden_layer_size 64-128-256 ---------------#

# sql_nemb 5 data_nemb 10 model structure [130, 32, 32, 1]
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset diabetes --nfeat 850 --nfield 48 \
    --moe_hid_layer_len 64 --data_nemb 10 \
    --dropout 0.2 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "dnn_hidden64" 


