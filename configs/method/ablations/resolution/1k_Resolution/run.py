import os

os.system(
    "python Main.py --device 0 --target obj -np 1000 -io configs/io/io_ama.json -o /data/kaltheuner/ICCV_AMA_1k -m configs/method/fit.json"
)
