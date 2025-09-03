import os

os.system(
    "python Main.py --device 2 --target obj -np 5000 -io configs/io/io_ama.json -o /data/kaltheuner/ICCV_Reg -m /home/kaltheuner/TemporalSurface/configs/method/ablations/method/SingleResolutionREG/SingleResolution.json"
)
