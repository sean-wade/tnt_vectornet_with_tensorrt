# train from scratch.
###
 # @Author: zhanghao
 # @LastEditTime: 2023-06-28 19:15:50
 # @FilePath: /my_vectornet_github/scripts/train_scratch.sh
 # @LastEditors: zhanghao
 # @Description: 
### 
python tools/train_tnt.py \
-d /mnt/data/SGTrain/TRAJ_DATASET/EXP8_Heading_Diamond_DIM10_BALANCE_MINI \
-o work_dir/tnt/EXP8_Heading_Diamond_DIM10_BALANCE_MINI/AUG_NF6 \
-b 1024 -e 200 -lr 0.01 \
-M 50 -K 6 -nf 6 \
-aux
