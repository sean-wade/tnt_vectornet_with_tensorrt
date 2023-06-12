# train from scratch.
python tools/train_tnt.py \
-d /home/jovyan/zhdata/TRAJ/EXP8_Heading_Diamond_DIM10_BALANCE \
-o work_dir/tnt/EXP8_Heading_Diamond_DIM10_BALANCE \
-b 4096 -e 200 -lr 0.04 \
-nf -aux -om
