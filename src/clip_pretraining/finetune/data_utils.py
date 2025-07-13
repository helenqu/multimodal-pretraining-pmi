import os
import torch
import json
import glob
import collections
import random

import numpy as np
import pandas as pd
import math

from tqdm import tqdm
from PIL import Image
from pathlib import Path
import pdb
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Sampler

class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch


def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)

    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(image_encoder, device_ids=[x for x in range(torch.cuda.device_count())])
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            features = image_encoder(batch['images'].cuda())

            all_data['features'].append(features.cpu())

            for key, val in batch.items():
                if key == 'images':
                    continue
                if hasattr(val, 'cpu'):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(is_train, image_encoder, dataset, device):
    split = 'train' if is_train else 'val'
    dname = type(dataset).__name__
    if image_encoder.cache_dir is not None:
        cache_dir = f'{image_encoder.cache_dir}/{dname}/{split}'
        cached_files = glob.glob(f'{cache_dir}/*')
    if image_encoder.cache_dir is not None and len(cached_files) > 0:
        print(f'Getting features from {cache_dir}')
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(f'Did not find cached features at {cache_dir}. Building from scratch.')
        loader = dataset.train_loader if is_train else dataset.test_loader
        data = get_features_helper(image_encoder, loader, device)
        if image_encoder.cache_dir is None:
            print('Not caching because no cache directory was passed.')
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Caching data at {cache_dir}')
            for name, val in data.items():
                torch.save(val, f'{cache_dir}/{name}.pt')
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device):
        self.data = get_features(is_train, image_encoder, dataset, device)

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data['features'] = torch.from_numpy(data['features']).float()
        return data


def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        print("getting dataloader")
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        dataloader = DataLoader(feature_dataset, batch_size=args.batch_size, shuffle=is_train)
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader

def filter_accessory_images(accessory_dir):
    threshold = 100
    width_threshold = 10
    height_threshold = 10

    if not (accessory_dir / "quality_scores_with_num_pixels_masked.csv").exists():
        quality_scores = pd.read_csv(accessory_dir / "quality_scores.csv")
        num_pixels_masked = []
        width = []
        height = []
        for i, row in quality_scores.iterrows():
            mask = np.load(row['image_path'].replace(".png", "_bg_mask.npy"))
            num_pixels_masked.append(np.sum(mask == 1))
            if np.sum(mask == 0) > threshold:
                left = np.min(np.where(~mask)[1])
                right = np.max(np.where(~mask)[1])
                top = np.min(np.where(~mask)[0]) 
                bottom = np.max(np.where(~mask)[0])
                width.append(right - left)
                height.append(bottom - top)
            else:
                width.append(0)
                height.append(0)
        quality_scores['num_pixels_masked'] = num_pixels_masked
        quality_scores['width'] = width
        quality_scores['height'] = height
        quality_scores.to_csv(accessory_dir / "quality_scores_with_num_pixels_masked.csv", index=False)
    else:
        quality_scores = pd.read_csv(accessory_dir / "quality_scores_with_num_pixels_masked.csv")
    
    quality_scores = quality_scores[
        (quality_scores['num_pixels_masked'] < (512**2)-threshold) & 
        (quality_scores['width'] > width_threshold) & 
        (quality_scores['height'] > height_threshold) & 
        (quality_scores['iou_prediction'] > 0.9)
    ]
    return quality_scores['image_path'].tolist()

def editing_transform(image, accessory_path):
    accessory_path = Path(accessory_path)
    accessory_image_mask = np.load(accessory_path.parent / f"{accessory_path.stem}_bg_mask.npy")
    accessory_image = np.array(Image.open(accessory_path).convert("RGBA"))
    alpha = np.ones_like(accessory_image_mask) * 255
    alpha[accessory_image_mask == 1] = 0
    accessory_image[:, :, 3] = alpha
    accessory_image = Image.fromarray(accessory_image)
    
    # Find smallest index in any row where mask is false to get bounding box
    left = np.min(np.where(~accessory_image_mask)[1])
    right = np.max(np.where(~accessory_image_mask)[1])
    top = np.min(np.where(~accessory_image_mask)[0]) 
    bottom = np.max(np.where(~accessory_image_mask)[0])
    
    accessory_image = accessory_image.crop((left, top, right, bottom))

    # scale accessory image to be in proportion with bg image
    accessory_image_area = accessory_image.width * accessory_image.height
    image_area = image.width * image.height
    factor = math.sqrt(image_area / (10*accessory_image_area))
    accessory_image = accessory_image.resize((math.ceil(accessory_image.width * factor), math.ceil(accessory_image.height * factor)))

    # randomly paste accessory image on bg image
    if accessory_image.width >= image.width or accessory_image.height >= image.height:
        return image
    x = random.choice(range(0, image.width-accessory_image.width, 100))
    y = random.choice(range(0, image.height-accessory_image.height, 100))
    image.paste(accessory_image, (x, y), accessory_image)
    return image

def save_results(all_labels, all_preds, all_metadata, outpath):
    pred_probs, pred_classes = all_preds.topk(5, 1, True, True)
    pred_probs = torch.nn.functional.softmax(pred_probs, dim=1).cpu()

    colnames = ['image_id', 'target', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5']
    column_values = [[Path(x).name for x in all_metadata], 
        all_labels.cpu().tolist(), 
        pred_classes.cpu()[:,0].tolist(), 
        pred_classes.cpu()[:,1].tolist(), 
        pred_classes.cpu()[:,2].tolist(), 
        pred_classes.cpu()[:,3].tolist(), 
        pred_classes.cpu()[:,4].tolist(), 
        pred_probs[:,0].tolist(), 
        pred_probs[:,1].tolist(), 
        pred_probs[:,2].tolist(), 
        pred_probs[:,3].tolist(), 
        pred_probs[:,4].tolist()
    ]
    df = pd.DataFrame({k: v for k, v in zip(colnames, column_values)})
    df.to_csv(outpath, index=False)