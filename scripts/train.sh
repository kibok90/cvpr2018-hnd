if [ "$1" = "imagenet" ] && [ "$2" = "relabel" ]; then
    python train.py -gpu -data ImageNet -test -m RLB -bsize 5000 -nep 50 -nlrd 2 -nl 1 -rl 30
elif [ "$1" = "imagenet" ] && [ "$2" = "loo" ]; then
    python train.py -gpu -data ImageNet -test -m LOO -bsize 5000 -nep 50 -nlrd 2 -loo 1 -cw
elif [ "$1" = "imagenet" ] && [ "$2" = "td" ]; then
    python train.py -gpu -data ImageNet -test -m TD -bsize 5000 -nep 50 -nlrd 2 -ex 1
elif [ "$1" = "imagenet" ] && [ "$2" = "td+loo" ]; then
    python train.py -gpu -data ImageNet -test -m TD+LOO -bsize 5000 -nep 50 -nlrd 2 -tdname "TD_-1_1e+00_0e+00_1e-02_1e-02" -loo 1 -cw -relu -sm l
elif [ "$1" = "awa2" ] && [ "$2" = "relabel" ]; then
    python train.py -gpu -data AWA2 -test -m RLB -bsize 0 -nep 1000 -nlrd 0 -rl 15
elif [ "$1" = "awa2" ] && [ "$2" = "loo" ]; then
    python train.py -gpu -data AWA2 -test -m LOO -bsize 0 -nep 1000 -nlrd 0 -loo 1
elif [ "$1" = "awa2" ] && [ "$2" = "td" ]; then
    python train.py -gpu -data AWA2 -test -m TD -bsize 0 -nep 1000 -nlrd 0 -ex 1 -cw
elif [ "$1" = "awa2" ] && [ "$2" = "td+loo" ]; then
    python train.py -gpu -data AWA2 -test -m TD+LOO -bsize 0 -nep 1000 -nlrd 0 -tdname "TD_-1_1e+00_0e+00_cw_1e-02_1e-02" -loo 1 -sm n
elif [ "$1" = "cub" ] && [ "$2" = "relabel" ]; then
    python train.py -gpu -data CUB -test -m RLB -bsize 0 -nep 1000 -nlrd 0 -rl 20
elif [ "$1" = "cub" ] && [ "$2" = "loo" ]; then
    python train.py -gpu -data CUB -test -m LOO -bsize 0 -nep 1000 -nlrd 0 -loo 1
elif [ "$1" = "cub" ] && [ "$2" = "td" ]; then
    python train.py -gpu -data CUB -test -m TD -bsize 0 -nep 1000 -nlrd 0 -ex 1 -cw
elif [ "$1" = "cub" ] && [ "$2" = "td+loo" ]; then
    python train.py -gpu -data CUB -test -m TD+LOO -bsize 0 -nep 1000 -nlrd 0 -tdname "TD_-1_1e+00_0e+00_cw_1e-02_1e-02" -loo 1 -relu -sm n
else
    echo "Usage: sh train.sh {d} {m}"
    echo "{d} = imagenet, awa2, cub"
    echo "{m} = relabel, loo, td, td+loo"
fi
