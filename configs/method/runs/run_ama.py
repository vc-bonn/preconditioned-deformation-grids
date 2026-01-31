import os

os.system(
    "python Main.py --skip 0 --device 0 --target obj -np 5000 --directory_path /data/kaltheuner/preprocessed-data/AMA -o /data/kaltheuner/ICCV_AMA_P -m configs/method/fit.json --init_grid_resolution 512"
)
