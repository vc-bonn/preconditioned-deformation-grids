import os

os.system(
    "python Main.py --device 1 --target obj -np 5000 -io configs/io/io_ama.json -o /data/kaltheuner/ICCV_No_Smoothness -m configs/method/ablations/noSmoothness/NoSmoothness.json "
)
