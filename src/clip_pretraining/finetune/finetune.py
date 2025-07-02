import os
import pdb
import copy
import time
import tqdm
import wandb
from pathlib import Path
import sys

import torch

from .args import parse_arguments
from .data_utils import get_dataloader, maybe_dictionarize
from .eval import evaluate
from .modeling import ClassificationHead, ImageEncoder, ImageClassifier
from .utils import cosine_lr, torch_load, LabelSmoothing
from .data import Dataset as datasets

def finetune(args):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."

    if args.wandb:
        wandb_name = Path(args.save).name
        wandb.init(project="clip-finetuning", 
                   config=args, 
                   name=wandb_name)

    image_classifier = ImageClassifier.load(args.load)

    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        image_enc = image_classifier.image_encoder
        print_every = 100
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier

        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
        image_classifier.process_images = True
        print_every = 100
    
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        apply_editing_transform=args.apply_editing_transform,
        editing_transform_p=args.editing_transform_p,
    )
    num_batches = len(dataset.train_loader)

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    for epoch in range(args.epochs):
        if not args.freeze_encoder:
            def disable_cudnn_hook(module, inputs):
                if isinstance(module, torch.nn.Conv2d):
                    torch.backends.cudnn.enabled = False
            hook = model.module.image_encoder.model.visual.conv1.register_forward_pre_hook(disable_cudnn_hook)

        model.train()
        data_loader = get_dataloader(
            dataset, is_train=True, args=args, image_encoder=image_enc)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            logits = model(inputs)

            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            batch_time = time.time() - start_time

            if i % print_every == 0:
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
                if args.wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/data_time": data_time,
                        "train/batch_time": batch_time,
                        "train/lr": optimizer.param_groups[0]['lr'],
                    })
                print("debug: logged to wandb")
        
        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            hook.remove()
            image_classifier = model.module
        print("debug: removed hook")

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
            print('Saving model to', model_path)
            image_classifier.save(model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch+1}.pt')
            torch.save(optimizer.state_dict(), optim_path)

        # Evaluate
        args.current_epoch = epoch
        eval_results = evaluate(image_classifier, args)
        if args.wandb:
            wandb.log({k:v for k,v in eval_results.items() if 'top' in k})

    if args.save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)
