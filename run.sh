if [ -f train.log ]; then
  mv train.log train.log.$(date +%Y%m%d%H%M)
fi

set -xe

python3 train_model.py \
	--dataset meps_analysis \
	--model hi_lam \
	--graph hierarchical \
	--n_workers 2 \
	--batch_size 1 \
	--epochs 400 \
	--load saved_models/hi_lam-4x64-02_23_20-1604/min_val_loss.ckpt \
	--ar_steps 6 > train.log 2>&1 &
