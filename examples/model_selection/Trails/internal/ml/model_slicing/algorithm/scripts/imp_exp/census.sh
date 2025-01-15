
# --------------------- DNN --------------------------
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
     --dataset census --nfeat 540 --nfield 41 \
     --moe_hid_layer_len 16 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 3\
     --epoch 50 --batch_size 1024 --lr 0.002 \
     --seed 3047 --beta=0.001 --gamma=0.001 \
     --train_dir dnn_K16_ablation --report_freq 5

~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
     --dataset census --nfeat 540 --nfield 41 \
     --moe_hid_layer_len 16 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 3\
     --epoch 50 --batch_size 1024 --lr 0.002 \
     --seed 3047 --beta=0 --gamma=0.001 \
     --train_dir dnn_K16_w/o_imp --report_freq 5



~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
     --dataset census --nfeat 540 --nfield 41 \
     --moe_hid_layer_len 16 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 3\
     --epoch 50 --batch_size 1024 --lr 0.002 \
     --seed 3047 --beta=0.001 --gamma=0 \
     --train_dir dnn_K16_w/o_spa --report_freq 5
    

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
     --dataset census --nfeat 540 --nfield 41 \
     --moe_hid_layer_len 16 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 3\
     --epoch 50 --batch_size 1024 --lr 0.002 \
     --seed 3047 --beta=0 --gamma=0 \
     --train_dir dnn_K16_w/o_both --report_freq 5

    

# --------------------- AFN --------------------------


~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
     --dataset census --nfeat 540 --nfield 41 \
    --moe_hid_layer_len 16 --nhid 16 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 3\
     --epoch 50 --batch_size 1024 --lr 0.002 \
     --seed 3047 --beta=0.001 --gamma=0.001 \
     --train_dir afn_K16_ablation --report_freq 5

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
     --dataset census --nfeat 540 --nfield 41 \
    --moe_hid_layer_len 16 --nhid 16 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 3\
     --epoch 50 --batch_size 1024 --lr 0.002 \
     --seed 3047 --beta=0 --gamma=0.001 \
     --train_dir afn_K16_w/o_imp --report_freq 5



~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
     --dataset census --nfeat 540 --nfield 41 \
    --moe_hid_layer_len 16 --nhid 16 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 3\
     --epoch 50 --batch_size 1024 --lr 0.002 \
     --seed 3047 --beta=0.001 --gamma=0 \
     --train_dir afn_K16_w/o_spa --report_freq 5
    

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparsemax_vertical_sams --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
     --dataset census --nfeat 540 --nfield 41 \
    --moe_hid_layer_len 16 --nhid 16 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 3\
     --epoch 50 --batch_size 1024 --lr 0.002 \
     --seed 3047 --beta=0 --gamma=0 \
     --train_dir afn_K16_w/o_both --report_freq 5