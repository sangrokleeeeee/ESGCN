CUDA_VISIBLE_DEVICES=0 python main.py --dataset pems07 --horizon 12 --window_size 12 --norm_method z_score --step_type multi --step_cl -1 --optimizer adam --batch_size 30