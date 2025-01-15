# DNN
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparseMoE --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_baseline\
    --dataset census --nfeat 540 --nfield 41  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --dropout 0.1  \
    --K 16 --C 1 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_DNN  --report_freq 5



# CIN
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparseMoE --expert cin\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_baseline\
    --dataset census --nfeat 540 --nfield 41  \
    --nhid 16 --data_nemb 10  \
    --dropout 0.1  \
    --K 16 --C 1 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_CIN  --report_freq 5


# AFN
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparseMoE --expert afn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log_baseline\
    --dataset census --nfeat 540 --nfield 41  \
    --moe_hid_layer_len 32 --nhid 16 --data_nemb 10  \
    --dropout 0.1  \
    --K 16 --C 1 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023\
    --train_dir MoESparse_AFN  --report_freq 5


# ARMNet
~/anaconda3/bin/python3 main.py \
    --device cuda:1 --net sparseMoE --net armnet\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset census --nfeat 540 --nfield 41  \
    --moe_hid_layer_len 32 --nhid 8 --nhead 4  --data_nemb 10 \
    --dropout 0 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir MoESparse_ARMNet  --report_freq 5