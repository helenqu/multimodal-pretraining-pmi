import torch
from tqdm import tqdm

from open_clip import OPENAI_IMAGENET_TEMPLATES

from .data import Dataset as datasets
from .modeling import ClassificationHead, ImageEncoder, ImageClassifier


def get_zeroshot_classifier(args, encoder):
    assert args.template == 'openai', "Only openai template is supported for now"
    assert args.train_dataset is not None
    template = OPENAI_IMAGENET_TEMPLATES
    
    clip_model = encoder.model
    tokenizer = encoder.tokenizer

    logit_scale = clip_model.logit_scale
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        classnames=args.classnames # this defaults to IMAGENET_CLASSNAMES
    )
    device = args.device
    clip_model.eval()
    clip_model.to(device)

    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = tokenizer(texts).to(device) # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def eval(args):
    args.freeze_encoder = True
    if args.load is not None:
        classifier = ImageClassifier.load(args.load)
    else:
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
    
    evaluate(classifier, args)

    if args.save is not None:
        classifier.save(args.save)

