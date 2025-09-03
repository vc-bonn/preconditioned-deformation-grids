import os

os.system(
    "python Main.py --skip 31 --device 4 --target obj -np 5000 -i tetra -k middle --directory_path /data/kaltheuner/preprocessed-data/AMA -o /data/kaltheuner/ICCV_AMA_P -m configs/method/fit.json "
)
