import argparse
import json
import os

from PIL import Image
import torch
import torchvision.transforms as transforms

from utils import get_model, load_model


def build_transform(image_size, crop_size):
    return transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.CenterCrop(size=crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_class_maps(metadata_dir):
    if not metadata_dir:
        return None, None

    class_idx_to_species_id_path = os.path.join(metadata_dir, 'class_idx_to_species_id.json')
    species_id_to_name_path = os.path.join(metadata_dir, 'plantnet300K_species_id_2_name.json')

    if not (os.path.exists(class_idx_to_species_id_path) and os.path.exists(species_id_to_name_path)):
        return None, None

    with open(class_idx_to_species_id_path, 'r', encoding='utf-8') as f:
        class_idx_to_species_id = json.load(f)

    with open(species_id_to_name_path, 'r', encoding='utf-8') as f:
        species_id_to_name = json.load(f)

    return class_idx_to_species_id, species_id_to_name


def resolve_label(class_index, class_idx_to_species_id, species_id_to_name):
    if class_idx_to_species_id is None or species_id_to_name is None:
        return None

    species_id = class_idx_to_species_id.get(str(class_index), class_idx_to_species_id.get(class_index))
    if species_id is None:
        return None

    return species_id_to_name.get(str(species_id), species_id_to_name.get(species_id, species_id))


def predict(image_path, checkpoint_path, model_name, num_classes, use_gpu, image_size, crop_size, topk, metadata_dir):
    device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')

    model_args = argparse.Namespace(model=model_name, pretrained=False)
    model = get_model(model_args, n_classes=num_classes)
    load_model(model, checkpoint_path, use_gpu=use_gpu and torch.cuda.is_available())
    model.to(device)
    model.eval()

    transform = build_transform(image_size=image_size, crop_size=crop_size)
    image = Image.open(image_path).convert('RGB')
    batch = transform(image).unsqueeze(0).to(device)

    class_idx_to_species_id, species_id_to_name = load_class_maps(metadata_dir)

    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[0]
        scores, indices = torch.topk(probabilities, k=topk)

    print(f'Image: {image_path}')
    print('Top predictions:')
    for rank, (score, class_index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        label = resolve_label(class_index, class_idx_to_species_id, species_id_to_name)
        if label is None:
            print(f'{rank}. class {class_index} - {score:.4f}')
        else:
            print(f'{rank}. class {class_index} - {label} - {score:.4f}')


def main():
    parser = argparse.ArgumentParser(description='Run a simple plant recognition demo on one image.')
    parser.add_argument('image_path', help='path to the image file')
    parser.add_argument('--checkpoint', required=True, help='path to the trained checkpoint (.tar)')
    parser.add_argument('--model', default='resnet18', help='model architecture used during training')
    parser.add_argument('--num_classes', type=int, default=1081, help='number of output classes')
    parser.add_argument('--topk', type=int, default=5, help='number of predictions to print')
    parser.add_argument('--image_size', type=int, default=256, help='resize shorter side to this size')
    parser.add_argument('--crop_size', type=int, default=224, help='center crop size')
    parser.add_argument('--cpu', action='store_true', help='force CPU inference')
    parser.add_argument('--metadata_dir', help='directory containing class_idx_to_species_id.json and species name map')
    args = parser.parse_args()

    use_gpu = not args.cpu
    predict(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        num_classes=args.num_classes,
        use_gpu=use_gpu,
        image_size=args.image_size,
        crop_size=args.crop_size,
        topk=args.topk,
        metadata_dir=args.metadata_dir,
    )


if __name__ == '__main__':
    main()