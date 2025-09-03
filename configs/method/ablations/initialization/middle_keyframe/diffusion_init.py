import os

os.system(
    "python Main.py --skip 0 --device 0 --target obj -np 5000 -i diffusion -k middle --directory_path /data/kaltheuner/DT4D2 -o /data/kaltheuner/ICCV_dt4d_P -m configs/method/fit.json "
)
