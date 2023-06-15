import torch.utils.data
import torchvision


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset


def build_dataset(image_set, args):
    if args.dataset_file == "hyper_spec":
        return build_hyper(image_set, args)
    if args.dataset_file == "hyper_seg":
        from .hyper_dataloader import buildHyperSeg

        return buildHyperSeg(image_set, args)
    raise ValueError(f"dataset {args.dataset_file} not supported")
