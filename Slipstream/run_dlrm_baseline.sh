python dlrm_baseline.py --arch-sparse-feature-size=16 \
			--arch-mlp-bot="13-512-256-64-16" \
			--arch-mlp-top="512-256-1" \
			--data-generation=dataset \
			--data-set=kaggle \
			--raw-data-file=./input/kaggle/train.txt \
			--processed-data-file=./input/kaggle/kaggleAdDisplayChallenge_processed.npz \
			--loss-function=bce \
			--round-targets=True \
			--mini-batch-size=1024 \
			--print-freq=4096 \
			--print-time

