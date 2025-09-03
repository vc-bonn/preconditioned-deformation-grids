import os

os.system(
    "python Main.py --device 7 --target obj -np 5000 -io configs/method/ablations/length/60steps/io_ama.json -o /data/kaltheuner/ICCV_AMA_60_Steps -m configs/method/ablations/length/60steps/fit.json"
)
