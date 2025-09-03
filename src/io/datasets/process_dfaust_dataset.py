import os
import shutil

if __name__ == "__main__":

    ama_path = "/data/kaltheuner/preprocessed-data/DFAUST"
    out_path = "/data/kaltheuner/processed_data/DFAUST"
    os.makedirs(out_path, exist_ok=True)

    lengths = [20]

    directories = [
        os.path.join(ama_path, d)
        for d in os.listdir(ama_path)
        if os.path.isdir(os.path.join(ama_path, d))
    ]
    numbers = list(set([
        int(d.split("_")[0])
        for d in os.listdir(ama_path)
        if os.path.isdir(os.path.join(ama_path, d))
    ]))
    for n in numbers:
        dirs = [d for d in directories if str(n) in d]
        dirs.sort()
        obj_files = []
        for d in dirs:
            objs = [os.path.join(d,"gt",f) for f in os.listdir(os.path.join(d,"gt")) if f[-4:] == ".obj"]
            objs.sort()
            obj_files += objs
            obj =  "_".join(obj_files[0].split("/")[-3].split("_")[:-1])
            for l in lengths:
                pos_idx = 0
                while (len(obj_files) - pos_idx) // l > 0:
                    seq_path = os.path.join(out_path, str(l), obj + "_" + str(pos_idx))
                    os.makedirs(seq_path, exist_ok=True)
                    seq_files = obj_files[pos_idx : pos_idx + l]
                    for f in seq_files:
                        shutil.copyfile(
                            f, os.path.join(seq_path, f.split("/")[-1])
                        )
                    pos_idx += l
