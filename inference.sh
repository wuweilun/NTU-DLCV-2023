# Origin 
python3 test.py --model 7B --max_seq_len 128 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset star --blr 9e-2 --weight_decay 0.16 --output_dir ./ --accum_iter 1 --vaq --qav --resume ./data/star/star.pth --filename origin

# llama epoch6
python3 test.py --model 7B --max_seq_len 128 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset star --blr 9e-2 --weight_decay 0.16 --output_dir ./ --accum_iter 1 --vaq --qav --resume checkpoint_best_llama_epoch6.pth --filename llama_epoch6

# Voting light
python3 voting_light.py

# llama epoch10
python3 test.py --model 7B --max_seq_len 128 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset star --blr 9e-2 --weight_decay 0.16 --output_dir ./output_dir --accum_iter 1 --vaq --qav --resume checkpoint_best_llama_epoch10.pth --filename llama_epoch10

# hint version
python3 test.py --model 7B --max_seq_len 228 --batch_size 1 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset star --blr 9e-2 --weight_decay 0.16 --output_dir ./output_dir --accum_iter 8 --vaq --qav --resume 'checkpoint_best_hint.pth' --filename hint_5277 --hint_data hint.json

# Voting 
python3 voting.py output_dir