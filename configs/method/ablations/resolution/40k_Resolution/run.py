import os

os.system(
    "python Main.py --device 5 --target obj -np 40000 -io configs/io/io_ama.json -o /data/kaltheuner/ICCV_AMA_40k -m configs/method/fit.json"
)
