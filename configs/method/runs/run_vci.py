import os

os.system(
    "python Main.py --skip 0 --device 0 --target obj -np 5000 --directory_path /data/kaltheuner/vci_data_meshes/processed_sequences -o /data/kaltheuner/ICCV_VCI -m configs/method/fit.json "
)
