import os

os.system(
    "python Main.py --device 0 --target obj -np 5000 -io configs/method/ablations/length/80steps/io_ama.json -o /data/kaltheuner/ICCV_AMA_80_Steps -m configs/method/ablations/length/80steps/fit.json"
)
