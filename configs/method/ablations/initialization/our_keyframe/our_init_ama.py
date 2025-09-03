import os

os.system(
    "python Main.py --skip 19 --device 7 --target obj -np 5000 -i ours -k ours --directory_path /data/kaltheuner/preprocessed-data/AMA -o /data/kaltheuner/ICCV_AMA_P -m configs/method/fit.json "
)
