import os

os.system(
    "python Main.py --device 3 --target obj -np 5000 -io configs/method/ablations/length/10steps/io_ama.json -o /data/kaltheuner/ICCV_AMA_10_Steps -m configs/method/ablations/length/10steps/fit.json"
)
