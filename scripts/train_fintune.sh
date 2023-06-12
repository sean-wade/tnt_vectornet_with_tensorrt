# train fintune.
python tools/train_tnt.py \
-d /home/jovyan/zhdata/TRAJ/EXP8_Heading_Diamond_DIM10_BALANCE \
-o work_dir/tnt/EXP8_Heading_Diamond_DIM10_BALANCE \
-b 4096 -e 100 -lr 0.004 -luf 20 -ldr 0.9 \
-rm work_dir/tnt/EXP8_Heading_Diamond_DIM10_BALANCE/06_09_09_48/best_TNT.pth

# -om
# -aux