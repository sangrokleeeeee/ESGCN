CUDA_VISIBLE_DEVICES=1 python main.py --dataset pems04 --horizon 12 --window_size 12 --norm_method z_score --step_type multi --step_cl -1 --optimizer adam --exponential_decay_step 5 --epoch 50 --batch_size 64