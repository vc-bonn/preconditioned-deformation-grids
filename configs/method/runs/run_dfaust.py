import os

os.system(
    "python Main.py --skip 0 --device 0 --target obj -np 5000 --directory_path /data/kaltheuner/preprocessed-data/DFAUST -o /data/kaltheuner/ICCV_dfaust_P -m configs/method/fit.json "
)
