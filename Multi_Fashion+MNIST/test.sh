dataroot=/home/tuantran/CAGrad/Ada_CAGrad/data/ # folder root data
weight=equal
dataname=multi_fashion_and_mnist # multi_mnist / multi_fashion / multi_fashion_and_mnist
method=graddrop # mgd / cagrad / pcgrad / adacagrad/ graddrop
alpha=0.2
bs=256
seed=0
python3 -u test.py --dataname $dataname --bs $bs --dataroot $dataroot --seed $seed --weight $weight --method $method --alpha $alpha 