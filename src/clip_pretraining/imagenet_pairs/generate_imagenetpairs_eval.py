# randomly choose a picture from IN val set for each class

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import random
import shutil
from tqdm import tqdm
import argparse

from clip_pretraining.finetune.data_utils import filter_accessory_images, accessory_transform

def name_from_path(x):
    return Path(x).name.split(".")[0]

def main(args):
    random.seed(args.seed)

    data_dir = Path("data/imagenet/val")
    outdir_root = Path("data/finetune_eval")
    
    accessory_dir = Path("data/accessory_images")
    if not (accessory_dir / "eval_split.txt").exists():
        all_accessory_paths = filter_accessory_images(accessory_dir)
        train_split = random.sample(all_accessory_paths, int(0.9 * len(all_accessory_paths)))
        eval_split = [x for x in all_accessory_paths if x not in train_split] + random.sample(all_accessory_paths, int(0.1 * len(all_accessory_paths)))
        print(f"{len(train_split)} train, {len(eval_split)} eval; {len(all_accessory_paths)} total")
        with open(accessory_dir / "train_split.txt", "w") as f:
            for path in train_split:
                f.write(f"{path}\n")
        with open(accessory_dir / "eval_split.txt", "w") as f:
            for path in eval_split:
                f.write(f"{path}\n")
        
        accessory_paths = eval_split
    else:
        accessory_paths = [Path(x) for x in open(accessory_dir / "eval_split.txt").readlines()]
    accessory_words = [name_from_path(x) for x in accessory_paths]
    print(f"using {len(accessory_paths)} accessory images")

    nonzero_freq_pairs = pd.read_csv("data/pair_to_imagenet_label/pairs.csv")
    nonzero_freq_pairs['other_word'] = [x.split(",")[0] for x in nonzero_freq_pairs['word_pair_final']]
    
    outdir = outdir_root / "imagenet_accessory_eval"
    print(f"writing accessory images to {outdir}")

    if outdir.exists():
        max_idx = max([int(x.name) for x in outdir.glob("*") if x.is_dir()]) if len(list(outdir.glob("*"))) > 0 else 0
        print(f"restarting at {max_idx}")
    else:
        max_idx = 0

    class_dirs = sorted([x for x in data_dir.glob("*") if x.is_dir()])

    for i, class_dir in tqdm(enumerate(class_dirs)):

        if i < max_idx:
            continue
        class_outdir = outdir / str(i).zfill(3)
        if class_outdir.exists():
            shutil.rmtree(class_outdir)
        class_outdir.mkdir(parents=True)

        nonzero_freq_pairs_for_class = nonzero_freq_pairs[(nonzero_freq_pairs['assigned_label_int'] == i) & (nonzero_freq_pairs['other_word'].isin(accessory_words))]
        nonzero_freq_words_for_class = nonzero_freq_pairs_for_class['other_word'].unique()

        total_images_for_class = len(list(class_dir.glob("*.JPEG")))
        if len(nonzero_freq_words_for_class) > 0:
            zero_freq_words_for_class = [x for x in nonzero_freq_words_for_class if x not in nonzero_freq_words_for_class]
            high_freq_words_for_class = nonzero_freq_pairs_for_class[nonzero_freq_pairs_for_class['frequency'] > 5000]['other_word'].unique()
            # take 10 very high freq words
            accessory_words_for_class = random.choices(high_freq_words_for_class, k=20) if len(high_freq_words_for_class) > 0 else []
            # take 5 random zero freq words
            accessory_words_for_class += random.choices(zero_freq_words_for_class, k=5) if len(zero_freq_words_for_class) > 0 else []
            # take random nonzero freq words, weighted by log frequency
            accessory_words_for_class += random.choices(nonzero_freq_words_for_class, weights=np.log10(nonzero_freq_pairs_for_class['frequency'].values), k=total_images_for_class - len(accessory_words_for_class))
        else:
            accessory_words_for_class = random.choices(accessory_words, k=total_images_for_class)
        accessory_paths_for_class = [accessory_dir / f"{x}.png" for x in accessory_words_for_class]

        for j, image_path in enumerate(class_dir.glob("*.JPEG")):
            image = Image.open(image_path)
            accessory_path = accessory_paths_for_class[j]
            accessoryed = accessory_transform(image, accessory_path)

            accessory_name = name_from_path(accessory_path)
            accessoryed.save(class_outdir / f"{accessory_name}_{name_from_path(image_path)}.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
