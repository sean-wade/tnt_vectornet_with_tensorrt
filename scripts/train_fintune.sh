# train fintune.
python tools/train_tnt.py \
-d /mnt/data/SGTrain/TRAJ_DATASET/EXP8_Heading_Diamond_DIM10_BALANCE_MINI \
-o work_dir/tnt/EXP8_Heading_Diamond_DIM10_BALANCE_MINI/AUG_NF6 \
-b 1024 -e 100 -lr 0.008 -luf 25 -ldr 0.9 \
-rm work_dir/tnt/EXP8_Heading_Diamond_DIM10_BALANCE_MINI/AUG_NF6/06_28_19_16/best_TNT.pth \
-M 50 -K 6 -nf 6 \
# -w 4
# -aux
# -om
# -aug

# python tools/train_tnt.py \
# -d /mnt/data/SGTrain/TRAJ_DATASET/EXP8_Heading_Diamond_DIM10_BALANCE \
# -o work_dir/tnt/EXP8_Heading_Diamond_DIM10_BALANCE/AUG2 \
# -b 1024 -e 100 -lr 0.004 -luf 20 -ldr 0.9 \
# -rm work_dir/tnt/EXP8_Heading_Diamond_DIM10_BALANCE/baseline/best_TNT.pth \
# -M 50 -K 6 -nf 10 \
# -w 4