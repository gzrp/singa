nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset movielens --nfeat 92000 --nfield 3  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Default:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&



nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset movielens --nfeat 92000 --nfield 3  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 200 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Epoch 200:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&



nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset movielens --nfeat 92000 --nfield 3  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 300 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Epoch 300:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

