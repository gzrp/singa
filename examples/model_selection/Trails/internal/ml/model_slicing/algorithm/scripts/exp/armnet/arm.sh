# census
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparsemax_vertical_sams --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
     --dataset census --nfeat 540 --nfield 41 \
    --moe_hid_layer_len 16 --nhid 8 --nhead 4 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 4\
     --epoch 30 --batch_size 1024 --lr 0.01 \
     --seed 3047 --beta=0.001 --gamma=0.003 \
     --train_dir armnet_K16 --report_freq 5

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparsemax_vertical_sams --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
     --dataset census --nfeat 540 --nfield 41 \
    --moe_hid_layer_len 16 --nhid 16 --nhead 16 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 4\
     --epoch 30 --batch_size 1024 --lr 0.01 \
     --seed 3047 --beta=0.001 --gamma=0.001 \
     --train_dir armnet_K16_test --report_freq 5

# hcdr
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net sparsemax_vertical_sams --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset hcdr --nfeat 550 --nfield 69\
    --moe_hid_layer_len 16 --nhid 8 --nhead 4 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 4\
     --epoch 30 --batch_size 1024 --lr 0.01 \
     --seed 3047 --beta=0.001 --gamma=0.003 \
     --train_dir armnet_K16 --report_freq 5

# diabetes
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparsemax_vertical_sams --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset diabetes --nfeat 850 --nfield 48 \
    --moe_hid_layer_len 16 --nhid 8 --nhead 4 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.2   \
     --K 16 --alpha 2 --max_filter_col 3\
     --epoch 30 --batch_size 1024 --lr 0.005 \
     --seed 3047 --beta=0.001 --gamma=0.004 \
     --train_dir armnet_K16 --report_freq 5


# credit
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert armnet\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset credit --nfeat 350 --nfield 23 \
    --moe_hid_layer_len 16 --nhid 8 --nhead 4 --data_nemb 10  \
     --hid_layer_len 16 --sql_nemb 10\
     --dropout 0.1   \
     --K 16 --alpha 2 --max_filter_col 3\
     --epoch 30 --batch_size 1024 --lr 0.005 \
     --seed 3047 --beta=0.001 --gamma=0.004 \
     --train_dir armnet_K16 --report_freq 5