import data.transforms as T
from torch.utils.data import DataLoader
from data.datasets import Market1501, CUHK03, DukeMTMCreID, MSMT17
from data.dataset_loader import ImageDataset
from data.samplers import RandomIdentitySampler


__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
}


def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __factory.keys()))

    print("Initializing dataset {}".format(config.DATA.DATASET))
    dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT, split_id=config.DATA.SPLIT_ID,
                                             cuhk03_labeled=config.DATA.CUHK03_LABELED, 
                                             cuhk03_classic_split=config.DATA.CUHK03_CLASSIC_SPLIT)

    return dataset


def build_transforms(config):
    transform_train = T.Compose([
        T.RandomCroping(config.DATA.HEIGHT, config.DATA.WIDTH, p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability = config.AUG.RE_PROB)
    ])

    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test


def build_dataloader(config):
    dataset = build_dataset(config)
    transform_train, transform_test = build_transforms(config)

    trainloader = DataLoader(ImageDataset(dataset.train, transform=transform_train),
                             sampler=RandomIdentitySampler(dataset.train, num_instances=config.DATA.NUM_INSTANCES),
                             batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=True, drop_last=True)
    queryloader = DataLoader(ImageDataset(dataset.query, transform=transform_test),
                             batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=True, drop_last=False, shuffle=False)

    galleryloader = DataLoader(ImageDataset(dataset.gallery, transform=transform_test),
                               batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                               pin_memory=True, drop_last=False, shuffle=False)

    return trainloader, queryloader, galleryloader, dataset.num_train_pids
