import os
import shutil

if __name__ == "__main__":

    ama_path = "/data/kaltheuner/datasets/AMA"
    out_path = "/data/kaltheuner/processed_data/AMA"
    os.makedirs(out_path, exist_ok=True)

    lengths = [10, 20, 40, 60, 80]

    directories = [
        os.path.join(ama_path, d)
        for d in os.listdir(ama_path)
        if os.path.isdir(os.path.join(ama_path, d))
    ]
    for dir_path in directories:
        obj_files = [f for f in os.listdir(dir_path) if f[-4:] == ".obj"]
        obj_files.sort()
        obj = dir_path.split("/")[-1]
        for l in lengths:
            pos_idx = 0
            while (len(obj_files) - pos_idx) // l > 0:
                seq_path = os.path.join(out_path, str(l), obj + "_" + str(pos_idx))
                os.makedirs(seq_path, exist_ok=True)
                seq_files = obj_files[pos_idx : pos_idx + l]
                for f in seq_files:
                    shutil.copyfile(
                        os.path.join(dir_path, f), os.path.join(seq_path, f)
                    )
                pos_idx += l
