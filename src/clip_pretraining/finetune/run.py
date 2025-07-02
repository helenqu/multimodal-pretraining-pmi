from pathlib import Path

from .args import parse_arguments
from .modeling import ImageEncoder, ClassificationHead, ImageClassifier
from .zeroshot import get_zeroshot_classifier
from .finetune import finetune
args = parse_arguments()

print("starting", flush=True)
if not args.load:
    image_encoder = ImageEncoder(args, keep_lang=True)
    classification_head = get_zeroshot_classifier(args, image_encoder)
    # delattr(image_encoder.model, 'transformer')
    classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
    zeroshot_checkpoint = Path(args.save) / 'zeroshot.pt'
    classifier.save(zeroshot_checkpoint)
else:
    zeroshot_checkpoint = Path(args.load)
    classifier = ImageClassifier.load(zeroshot_checkpoint)

# Standard fine-tuning
args.save = Path(args.save) / 'finetuned'
finetuned_checkpoint = finetune(args)
