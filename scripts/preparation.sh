if [ "$1" = "imagenet_full" ]; then
    python preparation.py
elif [ "$1" = "imagenet" ]; then
    python build_taxonomy.py ImageNet
elif [ "$1" = "awa2" ]; then
    python build_taxonomy.py AWA
    ln -s AWA taxonomy/AWA2
elif [ "$1" = "cub" ]; then
    python build_taxonomy.py CUB
fi
