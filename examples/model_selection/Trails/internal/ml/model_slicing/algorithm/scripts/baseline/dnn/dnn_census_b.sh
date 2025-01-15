

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset census --nfeat 540 --nfield 41  \
    --moe_hid_layer_len 64 --data_nemb 10 \
    --dropout 0 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "dnn_hidden32" 


~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset census --nfeat 540 --nfield 41  \
    --moe_hid_layer_len 16 --data_nemb 10 \
    --dropout 0.1 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "dnn_hidden16" 


~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset census --nfeat 540 --nfield 41  \
    --moe_hid_layer_len 8 --data_nemb 10 \
    --dropout 0.1 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "dnn_hidden16" 