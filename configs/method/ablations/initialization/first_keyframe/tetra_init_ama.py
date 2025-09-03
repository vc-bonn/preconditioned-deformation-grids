import os

os.system(
    "python Main.py --skip 4 --device 1 --target obj -np 5000 -i tetra -k first --directory_path /data/kaltheuner/preprocessed-data/AMA -o /data/kaltheuner/ICCV_AMA_P -m configs/method/fit.json "
)
