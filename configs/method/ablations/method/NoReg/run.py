import os

os.system(
    "python Main.py --device 0 --target obj -np 5000 -io configs/io/io_ama.json -o /data/kaltheuner/ICCV_AMA_No_Reg -m configs/method/ablations/NoReg/fit.json"
)
