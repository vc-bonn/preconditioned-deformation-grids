import os

os.system(
    "python Main.py --device 5 --target obj -np 20000 -io configs/method/ablations/resolution/20k_Resolution/io_ama.json -o /data/kaltheuner/ICCV_AMA_20k -m configs/method/fit.json"
)
