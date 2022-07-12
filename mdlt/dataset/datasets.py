import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, SVHN, VisionDataset
from torchvision.transforms.functional import rotate
from mdlt.utils.misc import save_image


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small MDLT datasets
    "ImbalancedColoredMNIST",
    "ImbalancedRotatedMNIST",
    "ImbalancedDigits",
    # Big MDLT datasets
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet"
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    MANY_SHOT_THRES = 100    # Default, subclasses may override
    FEW_SHOT_THRES = 20      # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MNISTM(VisionDataset):
    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]
    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        if not self._check_exists():
            raise RuntimeError("Dataset not found.")
        data_file = self.training_file if self.train else self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))


class ImbalancedMultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, split, environments, dataset_transform, input_shape, num_classes,
                 imb_type_per_env, imb_factor=0.1, rand_seed=0):
        super().__init__()
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images_tr = original_dataset_tr.data
        original_images_te = original_dataset_te.data
        original_labels_tr = original_dataset_tr.targets
        original_labels_te = original_dataset_te.targets

        shuffle_tr = torch.randperm(len(original_images_tr))
        original_images_tr = original_images_tr[shuffle_tr]
        original_labels_tr = original_labels_tr[shuffle_tr]
        # split original train into train & val set
        val_size = len(original_images_tr) // 3
        original_images_va = original_images_tr[:val_size]
        original_labels_va = original_labels_tr[:val_size]
        original_images_tr = original_images_tr[val_size:]
        original_labels_tr = original_labels_tr[val_size:]

        shuffle_te = torch.randperm(len(original_images_te))
        original_images_te = original_images_te[shuffle_te]
        original_labels_te = original_labels_te[shuffle_te]

        if split == 'train':
            original_images = original_images_tr
            original_labels = original_labels_tr
        elif split == 'val':
            original_images = original_images_va
            original_labels = original_labels_va
        else:
            original_images = original_images_te
            original_labels = original_labels_te

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            if split == 'train':
                img_max = 1000
                img_num_list = self.get_img_num_per_cls(num_classes, img_max, imb_type_per_env[i], imb_factor)
            else:
                img_max = 500
                img_num_list = self.get_img_num_per_cls(num_classes, img_max, 'balanced', 1.)
            images, labels = self.gen_imbalanced_data(images, labels, img_num_list)
            self.datasets.append(dataset_transform(images, labels, environments[i]))

    @staticmethod
    def get_img_num_per_cls(cls_num, img_max, imb_type, imb_factor):
        img_num_per_cls = []
        if 'exp' in imb_type:
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.)))
                img_num_per_cls.append(int(num))
        elif 'step' in imb_type:
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        if 'inv' in imb_type:
            img_num_per_cls = img_num_per_cls[::-1]
        return img_num_per_cls

    def gen_imbalanced_data(self, images, labels, img_num_per_cls):
        new_images, new_labels = [], []
        labels_np = np.array(labels, dtype=np.int64)
        classes = np.unique(labels_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(labels_np == the_class)[0]
            np.random.shuffle(idx)
            select_idx = idx[:the_img_num]
            new_images.append(images[select_idx, ...])
            new_labels.append(labels[select_idx])
        new_images = torch.vstack(new_images)
        new_labels = torch.hstack(new_labels)
        return new_images, new_labels

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.num_classes):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def collect_vis_data(self, images, labels, num_per_cls=2):
        new_images = []
        labels_np = np.array(labels, dtype=np.int64)
        for class_idx in range(self.num_classes):
            idx = np.where(labels_np == class_idx)[0]
            new_images.append(images[idx[:num_per_cls], ...])
        new_images = torch.vstack(new_images)
        return new_images


class ImbalancedColoredMNIST(ImbalancedMultipleEnvironmentMNIST):
    ENVIRONMENTS = ['blue', 'gray', 'green', 'pink']
    COLORS = torch.tensor(
        [[65., 105., 225.], [180., 180., 180.], [20., 180., 20.], [255., 20., 147.],
         [255., 215., 0.], [255., 0., 0.], [0., 255., 0.], [0., 225., 225.], [0., 0., 255.], [188., 143., 143.]])

    def __init__(self, root, split, hparams, vis=False):
        self.vis = vis
        super(ImbalancedColoredMNIST, self).__init__(
            root, split, [0, 1, 2, 3], self.color_dataset, (3, 28, 28,), 10,
            hparams['imb_type_per_env'], hparams['imb_factor'], hparams['rand_seed'])

    def color_dataset(self, images, labels, environment):
        images = torch.stack([images, images, images], dim=1)
        images = (images > 0).float()
        images *= self.COLORS[environment].view(1, -1, 1, 1)

        x = images.float().div_(255.)
        y = labels.view(-1).long()

        if self.vis:
            os.makedirs('vis_data', exist_ok=True)
            save_image(self.collect_vis_data(images, labels),
                       f"vis_data/colormnist_env_{self.ENVIRONMENTS[environment]}.png", nrow=10)

        return TensorDataset(x, y)

    @staticmethod
    def torch_bernoulli_(p, size):
        return (torch.rand(size) < p).float()

    @staticmethod
    def torch_xor_(a, b):
        return (a - b).abs()


class ImbalancedRotatedMNIST(ImbalancedMultipleEnvironmentMNIST):
    # ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']
    ENVIRONMENTS = ['0', '30', '60']

    def __init__(self, root, split, hparams, vis=False):
        self.vis = vis
        super(ImbalancedRotatedMNIST, self).__init__(
            root, split, [0, 30, 60], self.rotate_dataset, (1, 28, 28,), 10,
            hparams['imb_type_per_env'], hparams['imb_factor'], hparams['rand_seed'])

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])
        y = labels.view(-1).long()

        if self.vis:
            os.makedirs('vis_data', exist_ok=True)
            save_image(self.collect_vis_data(x * 255., labels), f"vis_data/rotmnist_env_{angle}.png", nrow=10)

        return TensorDataset(x, y)


class ImbalancedDigits(MultipleDomainDataset):
    ENVIRONMENTS = ['MNIST', 'MNIST-M', 'SVHN']
    DATASET_MAPPINGS = {'MNIST': MNIST, 'MNIST-M': MNISTM, 'SVHN': SVHN}

    def __init__(self, root, split, hparams, vis=False, input_shape=(3, 28, 28,), num_classes=10):
        super().__init__()
        np.random.seed(hparams['rand_seed'])
        torch.manual_seed(hparams['rand_seed'])
        if root is None:
            raise ValueError('Data directory not specified!')

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.vis = vis
        self.datasets = []

        for i, env in enumerate(self.ENVIRONMENTS):
            if env != 'SVHN':
                original_dataset_tr = self.DATASET_MAPPINGS[env](root, train=True, download=True)
                original_dataset_te = self.DATASET_MAPPINGS[env](root, train=False, download=True)
                images_tr, images_te = original_dataset_tr.data, original_dataset_te.data
                labels_tr, labels_te = original_dataset_tr.targets, original_dataset_te.targets
            else:
                original_dataset_tr = self.DATASET_MAPPINGS[env](os.path.join(root, env), split='train', download=True)
                original_dataset_te = self.DATASET_MAPPINGS[env](os.path.join(root, env), split='test', download=True)
                images_tr, images_te = torch.from_numpy(original_dataset_tr.data), torch.from_numpy(original_dataset_te.data)
                labels_tr, labels_te = torch.from_numpy(original_dataset_tr.labels), torch.from_numpy(original_dataset_te.labels)

            shuffle = torch.randperm(len(images_tr))
            images_tr, labels_tr = images_tr[shuffle], labels_tr[shuffle]
            # split original train into train & val set
            val_size = len(images_tr) // 3
            images_va, labels_va = images_tr[:val_size], labels_tr[:val_size]
            images_tr, labels_tr = images_tr[val_size:], labels_tr[val_size:]

            if split == 'train':
                images, labels = images_tr, labels_tr
                img_max = 1000
                img_num_list = self.get_img_num_per_cls(
                    num_classes, img_max, hparams['imb_type_per_env'][i], hparams['imb_factor'])
            elif split == 'val':
                images, labels = images_va, labels_va
                img_max = 800
                img_num_list = self.get_img_num_per_cls(num_classes, img_max, 'balanced', 1.)
            else:
                images, labels = images_te, labels_te
                img_max = 800
                img_num_list = self.get_img_num_per_cls(num_classes, img_max, 'balanced', 1.)

            images, labels = self.gen_imbalanced_data(images, labels, img_num_list, env == 'SVHN')
            self.datasets.append(self.digit_transform(images, labels, env))

    def digit_transform(self, images, labels, dataset):
        if dataset == 'MNIST':
            images = torch.stack([images, images, images], dim=1)
        elif dataset == 'MNIST-M':
            images = images.permute(0, 3, 1, 2)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.ToTensor()]) if dataset == 'SVHN' else \
            transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

        x = torch.zeros(len(images), *self.input_shape)
        for i in range(len(images)):
            x[i] = transform(images[i])
        y = labels.view(-1).long()

        if self.vis:
            os.makedirs('vis_data', exist_ok=True)
            save_image(self.collect_vis_data(x, labels), f"vis_data/digit_{dataset}.png", nrow=8)

        return TensorDataset(x, y)

    @staticmethod
    def get_img_num_per_cls(cls_num, img_max, imb_type, imb_factor):
        img_num_per_cls = []
        if 'exp' in imb_type:
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.)))
                img_num_per_cls.append(int(num))
        elif 'step' in imb_type:
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        if 'inv' in imb_type:
            img_num_per_cls = img_num_per_cls[::-1]
        return img_num_per_cls

    def gen_imbalanced_data(self, images, labels, img_num_per_cls, shift_class=False):
        new_images, new_labels = [], []
        labels_np = np.array(labels, dtype=np.int64)
        classes = np.unique(labels_np)
        # if shift_class:
        #     classes = np.concatenate([classes[1:], classes[:1]], axis=0)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(labels_np == the_class)[0]
            np.random.shuffle(idx)
            select_idx = idx[:the_img_num]
            new_images.append(images[select_idx, ...])
            new_labels.append(labels[select_idx])
        new_images = torch.vstack(new_images)
        new_labels = torch.hstack(new_labels)
        return new_images, new_labels

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.num_classes):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def collect_vis_data(self, images, labels, num_per_cls=4):
        new_images = []
        labels_np = np.array(labels, dtype=np.int64)
        for class_idx in range(self.num_classes):
            idx = np.where(labels_np == class_idx)[0]
            new_images.append(images[idx[:num_per_cls], ...])
        new_images = torch.vstack(new_images)
        return new_images


class SplitImageFolder(torch.utils.data.Dataset):

    def __init__(self, path, df, augment, split='train'):
        self.df = df[df['split'] == split]
        self.img_dir = path
        self.split = split
        self.augment = augment
        self.transform = self.get_transform()
        self.targets = self.df.label
        self.classes = sorted(list(set([x.split('/')[1] for x in df.path])))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.img_dir, row['path'])).convert('RGB')
        img = self.transform(img)
        y = row['label'].astype('int')
        return img, y

    def get_transform(self):
        if self.augment and self.split == 'train':
            transform = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transform


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, df, split, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        self.datasets = []
        for i, environment in enumerate(environments):
            df_env = df[df['env'] == environment]
            env_dataset = SplitImageFolder(root, df_env, augment=augment, split=split)
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["C", "L", "S", "V"]
    MANY_SHOT_THRES = 100
    FEW_SHOT_THRES = 20

    def __init__(self, root, split, hparams):
        self.dir = os.path.join(root, "VLCS")
        self.df = pd.read_csv(os.path.join(self.dir, "VLCS.csv"))
        super().__init__(self.dir, self.df, split, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["A", "C", "P", "S"]
    MANY_SHOT_THRES = 100
    FEW_SHOT_THRES = 20

    def __init__(self, root, split, hparams):
        self.dir = os.path.join(root, "PACS")
        self.df = pd.read_csv(os.path.join(self.dir, "PACS.csv"))
        super().__init__(self.dir, self.df, split, hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    N_STEPS = 15001
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    MANY_SHOT_THRES = 100
    FEW_SHOT_THRES = 20

    def __init__(self, root, split, hparams):
        self.dir = os.path.join(root, "domain_net")
        self.df = pd.read_csv(os.path.join(self.dir, "DomainNet.csv"))
        super().__init__(self.dir, self.df, split, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["A", "C", "P", "R"]
    MANY_SHOT_THRES = 60
    FEW_SHOT_THRES = 20

    def __init__(self, root, split, hparams):
        self.dir = os.path.join(root, "office_home")
        self.df = pd.read_csv(os.path.join(self.dir, "OfficeHome.csv"))
        super().__init__(self.dir, self.df, split, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    MANY_SHOT_THRES = 100
    FEW_SHOT_THRES = 25

    def __init__(self, root, split, hparams):
        self.dir = os.path.join(root, "terra_incognita")
        self.df = pd.read_csv(os.path.join(self.dir, "TerraIncognita.csv"))
        super().__init__(self.dir, self.df, split, hparams['data_augmentation'], hparams)
