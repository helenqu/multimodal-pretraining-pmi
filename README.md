# Impact of Pretraining Word Co-occurrence on Compositional Generalization in Multimodal Models
This repository is the official implementation of [Impact of Pretraining Word Co-occurrence on Compositional Generalization in Multimodal Models](https://arxiv.org/abs/2507.08000).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

You will also need to download:
- LAION-400M captions
- TextVQA
- VQAv2
- [GenPairs](https://huggingface.co/datasets/helenqu/GenPairs) (synthetic dataset of images generated from concept pairs)
- [ImageNet-Paste](https://huggingface.co/datasets/helenqu/ImageNet-Paste) (ImageNet validation set augmented by pasting in images of other concepts)

## Concept Pair Extraction and PMI Calculation

To extract concept pairs and frequencies from LAION-400M captions, run:

```bash
./scripts/preprocess_laion_captions.sh <path_to_laion_400m_captions>
```

To calculate PMI from a csv of concept pairs with frequencies, run:

```bash
python src/clip_pretraining/utils.py --pairs_csv <path_to_csv>
```

### VQA Pair Frequencies

- compute all pairs for all questions (e.g. TextVQA): 
```bash
python vlm_eval/vqa_pair_freqs.py \
    --questions data/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --annotations data/textvqa/TextVQA_0.5.1_val.json \
    --compute_pairs
```
- get freqs for all pairs from LAION-400M: 
```bash
./scripts/vqa_pair_freqs.sh <path_to_pairs_csv (output of previous step)>
```

- aggregate output of previous step into single file: 
```bash
python vlm_eval/vqa_pair_freqs.py \
    --questions data/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --process_freqs
```

## Datasets

### GenPairs Dataset
The GenPairs dataset used in our evaluations can be downloaded [here](https://huggingface.co/datasets/helenqu/GenPairs).
To create your own version of the GenPairs dataset, run:

```bash
./scripts/create_genpairs.sh <path_to_pairs_data> <output_path>
```

### ImageNet-Paste Dataset
The ImageNet-Paste dataset used in our evaluations can be downloaded [here](https://huggingface.co/datasets/helenqu/ImageNet-Paste).
To create your own version of the ImageNet-Paste dataset, run:

```bash
./scripts/create_imagenetpairs.sh
```
with the correct paths in `src/clip_pretraining/imagenet_pairs/autogenerate_accessory_images.py` and `src/clip_pretraining/imagenet_pairs/generate_imagenetpairs_eval.py`.

## Model Training and Evals

### Zero-shot Classification

To run zero-shot classification, run:

```bash
./scripts/zero_shot.sh <path_to_data> <OpenCLIP_model_name>
```

### Fine-tuning and Evals
The fine-tuning and eval code is adapted from [WiSE-FT](https://github.com/mlfoundations/wise-ft/tree/master).

To fine-tune a CLIP model on either ImageNet or edited ImageNet images, run:

```bash
./scripts/finetune.sh <OpenCLIP_model_name> <save_dir>
```

(the last two args in the script must be uncommented to apply editing transform)

To evaluate a finetuned model, run:

```bash
./scripts/eval_finetuned_model.sh <model_path> <data_dir>
```

### LLaVA-1.5 VQA Evals

Code and instructions for the fine-tuning the custom LLaVA-1.5 based on LAION-400M-pretrained CLIP and VQA evals are provided in [this repository](https://github.com/helenqu/llava-pretraining-pmi).

## Citation
If you found this repository helpful, please cite our paper!
```bib
@article{qu2025impact,
  title={Impact of Pretraining Word Co-occurrence on Compositional Generalization in Multimodal Models},
  author={Qu, Helen and Xie, Sang Michael},
  journal={arXiv preprint arXiv:2507.08000},
  year={2025}
}
```

