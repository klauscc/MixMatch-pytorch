name=`basename "$0"`
workspace=/home/fengchan/stor/workspace/courses/comp790-dl3d/mixmatch/$name
OMP_NUM_THREADS=8 python train_gmmmix.py \
    --gpu 7 --n-labeled 1000 \
    --out $workspace \
    --layers_mix 0 \
