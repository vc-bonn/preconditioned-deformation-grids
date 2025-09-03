import os

os.system(
    "python Main.py --device 3 --target obj -np 10000 -io configs/io/io_ama.json -o /data/kaltheuner/ICCV_AMA_10k -m configs/method/fit.json"
)
