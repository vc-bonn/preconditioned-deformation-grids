import os

os.system(
    "python Main.py --device 2 --target obj -np 5000 -io configs/method/ablations/length/40steps/io_ama.json -o /data/kaltheuner/ICCV_AMA_40_Steps -m configs/method/ablations/length/40steps/fit.json"
)
