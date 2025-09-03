import os

os.system(
    "python Main.py --device 6 --target obj -np 5000 -io configs/method/ablations/length/20steps/io_ama.json -o /data/kaltheuner/ICCV_AMA_20_Steps -m configs/method/ablations/length/20steps/fit.json"
)
