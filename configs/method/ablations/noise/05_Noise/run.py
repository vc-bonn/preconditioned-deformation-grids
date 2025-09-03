import os

os.system(
    "python Main.py --device 0 --target obj -np 5000 -io configs/method/ablations/noise/05_Noise/io_ama.json -o /data/kaltheuner/ICCV_AMA_05_Noise -m configs/method/fit.json"
)
