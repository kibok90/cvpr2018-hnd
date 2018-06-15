if [ "$1" = "imagenet" ]; then
    if [ ! -f "resnet101-5d3b4d8f.pth" ]; then
        wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    fi
    python utils.py
elif [ "$1" = "awa2" ]; then
    python split_data.py AWA2
elif [ "$1" = "cub" ]; then
    python split_data.py CUB
fi
