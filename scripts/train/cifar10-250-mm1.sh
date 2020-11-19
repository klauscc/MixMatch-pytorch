name=`basename "$0"`
OMP_NUM_THREADS=8 python train.py \
    --gpu 2 --n-labeled 250 \
    --out /home/fengchan/stor/workspace/courses/comp790-dl3d/mixmatch/$name \
    --layers_mix 1 \
