import open_clip
from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES
from open_clip_train.precision import get_autocast

import torch
import torchvision.datasets as datasets

from tqdm import tqdm
import argparse
import logging
#TODO: logging isn't going to stdout or stderr??
from pathlib import Path
import pandas as pd
import os

RESULTS_COLNAMES = ['image_id', 'target', 'pred1', 'pred2', 'pred3', 'pred4', 'pred5', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5']

class ImageFolderWithPaths(datasets.ImageFolder):
    # Override __getitem__ to return image, label, and path
    def __getitem__(self, index):
        # This is the standard ImageFolder behavior
        path, label = self.samples[index]
        image = self.loader(path)
        # check if Path(path).stem is numeric with string methods

        image_id = int(Path(path).stem) if Path(path).stem.isdigit() else Path(path).stem
        #int(Path(path).stem.split('_')[-1])
        if self.transform is not None:
            image = self.transform(image)

        # Return the image, label, and path (or filename)
        return image, label, image_id
    
    def find_classes(self, directory: str):
        classes = sorted([entry.name for entry in os.scandir(directory) if entry.is_dir()])
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        if len(classes) == 1000:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx
        # if not all imagenet classes are present, want to preserve the original class indices
        class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
        return classes, class_to_idx

def accuracy(output, target, topk=(1,)):
    pred_values, pred_classes = output.topk(max(topk), 1, True, True)
    pred_classes = pred_classes.t()
    correct = pred_classes.eq(target.view(1, -1).expand_as(pred_classes))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def process_results(logits, target, ids):
    pred_probs, pred_classes = logits.topk(5, 1, True, True)
    pred_probs = torch.nn.functional.softmax(pred_probs, dim=1).cpu()
    # if len(ids.shape) == 1:
    #     ids = ids.unsqueeze(1) # want shape batch_size x 1
    # if len(target.shape) == 1:
    #     target = target.unsqueeze(1) # want shape batch_size x 1
    # new_rows = torch.cat([ids.cpu(), target.cpu(), pred_classes.cpu(), pred_probs], dim=1)
    column_values = [list(ids), 
        target.cpu().tolist(), 
        pred_classes.cpu()[:,0].tolist(), 
        pred_classes.cpu()[:,1].tolist(), 
        pred_classes.cpu()[:,2].tolist(), 
        pred_classes.cpu()[:,3].tolist(), 
        pred_classes.cpu()[:,4].tolist(), 
        pred_probs[:,0].tolist(), 
        pred_probs[:,1].tolist(), 
        pred_probs[:,2].tolist(), 
        pred_probs[:,3].tolist(), 
        pred_probs[:,4].tolist()]
    new_rows = pd.DataFrame({k: v for k, v in zip(RESULTS_COLNAMES, column_values)})

    # convert first example to probs & classnames for printing
    pred_classes = [IMAGENET_CLASSNAMES[i] for i in pred_classes[0].tolist()]
    pred_probs = pred_probs[0].tolist()
    target_str = IMAGENET_CLASSNAMES[target[0].item()]

    print(f"Target: {target_str}", flush=True)
    print("Predicted:", flush=True)
    for prob, cls in zip(pred_probs, pred_classes):
        print(f"{cls} ({prob:.2f})", flush=True)

    return new_rows

def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.
        all_results = pd.DataFrame()
        for i, (images, target, ids) in enumerate(tqdm(dataloader, unit_scale=args.batch_size)):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            results = process_results(logits, target, ids)
            # all_results = torch.cat([all_results, results], dim=0)
            all_results = pd.concat([all_results, results], ignore_index=True)
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
    
    # dtypes = {k: 'int' for k in RESULTS_COLNAMES[:7]}
    # dtypes.update({k: 'float' for k in RESULTS_COLNAMES[7:]})
    # results_df = pd.DataFrame(all_results.numpy(), columns=RESULTS_COLNAMES).astype(dtypes)
    top1 = (top1 / n)
    top5 = (top5 / n)

    return top1, top5, all_results


def zero_shot_eval(model, dataloader, args, tokenizer=None):
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    #TODO: remove after test
    cornbread_templates = [lambda c: f"{f(c)} and cornbread" for f in OPENAI_IMAGENET_TEMPLATES]
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            # templates=cornbread_templates,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}

    top1, top5, predictions = run(model, classifier, dataloader, args)
    results['imagenet-top1'] = top1
    results['imagenet-top5'] = top5

    # outpath = Path(args.data_dir) / 'predictions_cornbread_template.csv'
    outpath = Path(args.data_dir) / f"predictions_{args.model}.csv"
    predictions.to_csv(outpath, index=False)
    print(f"Saved predictions to {outpath}")

    logging.info('Finished zero-shot imagenet.')

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot evaluation')
    parser.add_argument('--model', default='ViT-B-32-quickgelu', type=str, help='model name')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--precision', default='fp32', type=str, help='precision')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--distributed', action='store_true', default=False, help='distributed')
    parser.add_argument('--horovod', action='store_true', default=False, help='horovod')
    args = parser.parse_args()
    print(vars(args), flush=True)

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)

    model_pretrained_map = {
        "ViT-B-32-quickgelu": "laion400m_e31",
        "ViT-B-16-plus-240": "laion400m_e32",
        "ViT-L-14": "laion400m_e32",
        "EVA01-g-14": "laion400m_s11b_b41k",
    }
    if args.model not in model_pretrained_map:
        raise ValueError(f"Model {args.model} not supported")

    model, _, preprocess_fn = open_clip.create_model_and_transforms(args.model, pretrained=model_pretrained_map[args.model])
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    model = model.to(device)

    tokenizer = open_clip.get_tokenizer(args.model)

    # dataset images saved in args.data_dir/class_id/pair_id.png
    # class_id = idx in IMAGENET_CLASSNAMES (can't do name since ImageFolder sorts the dirnames and uses that order)
    dataset = ImageFolderWithPaths(args.data_dir, transform=preprocess_fn)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=2, shuffle=False)

    zeroshot_results = zero_shot_eval(model, dataloader, args, tokenizer=tokenizer)
    print(f"RESULTS_STR_CLIP: {zeroshot_results}")