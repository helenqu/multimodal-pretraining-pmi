import os
import pdb
from pathlib import Path
import torch

from .data_utils import editing_transform
from .utils import num_correct
from ..utils import class_level_acc
from clip_pretraining.zero_shot_imagenet.run_zero_shot import ImageFolderWithPaths

from open_clip import IMAGENET_CLASSNAMES
import numpy as np
import pandas as pd
import random
import json

def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)

def save_results(ids, preds, labels, outpath):
    pred_probs, pred_classes = preds.topk(5, 1, True, True)
    pred_probs = torch.nn.functional.softmax(pred_probs, dim=1).cpu()

    colnames = ['image_id', 'target', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5']
    column_values = [ids,
        labels, 
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

    print(f"writing preds to {outpath}")
    df.to_csv(outpath, index=False)

    return df

def basic_save_results_postloop(all_labels, all_preds, all_metadata, args, data_dir):
    model_name = Path(args.model_path).name if "zeroshot" in args.model_path else Path(args.model_path).parent.parent.name
    outpath = data_dir / f"{model_name}_FT_predictions.csv"
    save_results([Path(x).name for x in all_metadata], all_preds, all_labels, outpath)
    return {}

class Dataset:
    class ImageNet:
        def __init__(self,
                    preprocess,
                    location=os.path.expanduser('~/data'),
                    batch_size=32,
                    num_workers=32,
                    classnames='openai',
                    debug=False,
                    apply_editing_transform=False,
                    editing_transform_p=None):
            self.preprocess = preprocess
            self.location = location
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.debug = debug
            self.apply_editing_transform = apply_editing_transform
            self.editing_transform_p = editing_transform_p
           
            accessory_dir = Path("data/accessory_images")
            if not (accessory_dir / "train_split.txt").exists():
                raise ValueError("train_split.txt does not exist")
            self.accessory_paths = [Path(x.strip()) for x in open(accessory_dir / "train_split.txt").readlines()]

            if classnames == 'openai':
                self.classnames = IMAGENET_CLASSNAMES
            else:
                raise ValueError(f"Invalid classnames: {classnames}")

            self.populate_train()
            self.populate_test()
        
        def _editing_transform(self, image):
            if random.random() < self.editing_transform_p:
                accessory_path = random.choice(self.accessory_paths)
                return self.preprocess(editing_transform(image, accessory_path))
            else:
                return self.preprocess(image)
        
        def populate_train(self):
            if self.debug:
                print("for debugging, populating with val instead")
                traindir = os.path.join(self.location, 'val')
            else:
                traindir = os.path.join(self.location, 'train')

            transform = self._editing_transform if self.apply_editing_transform else self.preprocess
            self.train_dataset = ImageFolderWithPaths(
                traindir,
                transform=transform)
            sampler = self.get_train_sampler()
            kwargs = {'shuffle' : True} if sampler is None else {}
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                sampler=sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                **kwargs,
            )

        def populate_test(self):
            self.test_dataset = self.get_test_dataset()
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=self.get_test_sampler()
            )

        def get_test_path(self):
            # test and train datasets are in different self.locations
            return Path(self.location) / self.name()

        def get_train_sampler(self):
            return None

        def get_test_sampler(self):
            return None

        def get_test_dataset(self):
            return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)
            

    class ImageNetTrain(ImageNet):
        def get_test_dataset(self):
            pass
    
    class ImageNetEval(ImageNet):
        def populate_train(self):
            pass

        def name(self):
            return "imagenet_val"
    
    class CustomDataset:
        def __init__(self,
                    preprocess,
                    location=os.path.expanduser('~/data'),
                    batch_size=32,
                    num_workers=32,
                    classnames='openai'):
            self.preprocess = preprocess
            self.location = location
            self.batch_size = batch_size
            self.num_workers = num_workers
            
            if classnames == 'openai':
                self.classnames = IMAGENET_CLASSNAMES
            else:
                raise ValueError(f"Invalid classnames: {classnames}")
            
            self.populate_train()
            self.populate_test()
        
        def populate_train(self):
            self.train_dataset = ImageFolderWithPaths(
                os.path.join(self.location, self.name()),
                transform=self.preprocess)
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )
        
        def populate_test(self):
            self.test_dataset = ImageFolderWithPaths(
                os.path.join(self.location, self.name()),
                transform=self.preprocess)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=self.get_test_sampler()
            )

        def get_test_sampler(self):
            return None
        
        def name(self):
            raise NotImplementedError()
    
    class CustomDatasetEval(CustomDataset):
        def populate_train(self):
            pass
    
    class FreqEval(CustomDatasetEval):
        def post_loop_metrics(self, all_labels, all_preds, all_metadata, args):
            if isinstance(all_metadata[0], torch.Tensor):
                ids = [x.item() for x in all_metadata]
            else:
                ids = [Path(x).name.split(".")[0] for x in all_metadata]

            outpath = Path(args.data_location) / self.name() / f"{Path(args.model_path).parent.parent.name}_FT_predictions.csv"
            df = save_results(ids, all_preds, all_labels.cpu().tolist(), outpath)

            return {"class-level accuracy": df[df['target'] == df['pred1']].shape[0] / df.shape[0]}
    
    class Freq0(FreqEval):
        def name(self):
            return "freq_0"

    class Freq1To2(FreqEval):
        def name(self):
            return "freq_1.00-2.93"

    class Freq1847To5411(FreqEval):
        def name(self):
            return "freq_1847.85-5411.70"

    class Freq215To630(FreqEval):
        def name(self):
            return "freq_215.44-630.96"

    class Freq25To73(FreqEval):
        def name(self):
            return "freq_25.12-73.56"

    class Freq2To8(FreqEval):
        def name(self):
            return "freq_2.93-8.58"

    class Freq5411To1100000(FreqEval):
        def name(self):
            return "freq_5411.70-1100000.00"

    class Freq630To1847(FreqEval):
        def name(self):
            return "freq_630.96-1847.85"

    class Freq73To215(FreqEval):
        def name(self):
            return "freq_73.56-215.44"

    class Freq8To25(FreqEval):
        def name(self):
            return "freq_8.58-25.12"
    
    class ZeroFreqTrain(CustomDataset):
        def populate_test(self):
            pass

        def name(self):
            return "zero_freq_train"

    class FreqAttack(CustomDatasetEval):
        def post_loop_metrics(self, all_labels, all_preds, all_metadata, args):
            outpath = Path(args.data_location) / self.name() / f"{Path(args.model_path).parent.parent.name}_FT_predictions.csv"
            df = save_results([Path(x).name.split(".")[0] for x in all_metadata], all_preds, all_labels.cpu().tolist(), outpath)

            return {"class-level accuracy": class_level_acc(df, "zero" in self.name())}