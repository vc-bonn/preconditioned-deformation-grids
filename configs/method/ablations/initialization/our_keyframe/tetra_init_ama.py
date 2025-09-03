import os

os.system(
    "python Main.py --skip 22 --device 6 --target obj -np 5000 -i tetra -k ours --directory_path /data/kaltheuner/preprocessed-data/AMA -o /data/kaltheuner/ICCV_AMA_P -m configs/method/fit.json "
)
