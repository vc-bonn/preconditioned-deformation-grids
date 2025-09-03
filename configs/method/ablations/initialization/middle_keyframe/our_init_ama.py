import os

os.system(
    "python Main.py --skip 12 --device 3 --target obj -np 5000 -i ours -k middle --directory_path /data/kaltheuner/preprocessed-data/AMA -o /data/kaltheuner/ICCV_AMA_P -m configs/method/fit.json "
)
