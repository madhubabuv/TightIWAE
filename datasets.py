from PIL import Image
from torchvision import datasets, transforms
import torch
import torch.utils.data as data
import urllib.request
import scipy.io
import os
import imageio
import numpy as np
from os import listdir
import os.path
import time

def load_binarised_MNIST(path, cuda, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    class stochMNIST(datasets.MNIST):
        """ Gets a new stochastic binarization of MNIST at each call. """

        def __getitem__(self, index):
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

            img = Image.fromarray(img.numpy(), mode='L')
            img = transforms.ToTensor()(img)
            img = torch.bernoulli(img)  # stochastically binarize
            return img, target

        def get_mean_img(self):
            imgs = self.train_data.type(torch.float) / 255
            mean_img = imgs.mean(0).reshape(-1).numpy()
            return mean_img

    train_loader = torch.utils.data.DataLoader(
        stochMNIST(path, train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size,
        shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        stochMNIST(path, train=False, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True,
        **kwargs)
    input_size = 784

    return train_loader, test_loader, input_size

def load_OMNIGLOT(path = "./datasets/omniglot", cuda=False, batch_size=20):
    """ Get from https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat """
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    class omniglot(data.Dataset):
        # data loader from https://github.com/yoonholee/pytorch-vae/blob/master/data_loader/omniglot.py
        """ omniglot dataset """
        url = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'

        def __init__(self, root, train=True, transform=None, download=False):
            # we ignore transform.
            self.root = os.path.expanduser(root)
            self.train = train  # training set or test set

            if download: self.download()
            if not self._check_exists():
                raise RuntimeError('Dataset not found. You can use download=True to download it')

            self.data = self._get_data(train=train)

        def __getitem__(self, index):
            img = self.data[index].reshape(28, 28)
            img = Image.fromarray(img)
            img = transforms.ToTensor()(img).type(torch.FloatTensor)
            img = torch.bernoulli(img)  # stochastically binarize
            return img, torch.tensor(-1)  # Meaningless tensor instead of target

        def __len__(self):
            return len(self.data)

        def _get_data(self, train=True):
            def reshape_data(data):
                return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='fortran')

            omni_raw = scipy.io.loadmat(os.path.join(self.root, 'chardata.mat'))
            data_str = 'data' if train else 'testdata'
            data = reshape_data(omni_raw[data_str].T.astype('float32'))
            return data

        def get_mean_img(self):
            return self.data.mean(0)

        def download(self):
            if self._check_exists():
                return
            if not os.path.exists(self.root):
                os.makedirs(self.root)

            print('Downloading from {}...'.format(self.url))
            local_filename = os.path.join(self.root, 'chardata.mat')
            urllib.request.urlretrieve(self.url, local_filename)
            print('Saved to {}'.format(local_filename))

        def _check_exists(self):
            return os.path.exists(os.path.join(self.root, 'chardata.mat'))

    train_loader = torch.utils.data.DataLoader(
        omniglot(path, train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size,
        shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        omniglot(path, train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size,
        shuffle=True, **kwargs)

    input_size = 784

    return train_loader, test_loader, input_size


def load_celeb_a_32x32(cuda, batch_size, path = "datasets/CelebA32x32/", train_test_split = 0.9):
    ## note, still needs some preprocessing ...
    # download and unpack from: https://drive.google.com/file/d/1eKq1RzppY6FYHqG1j1CWrWEvWNksZJSB/view?usp=sharing

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    baked_path = path+"celeba_32x32.npy"
    if os.path.isfile(baked_path):
        data = np.load(baked_path)
        print("data loaded as:", data.shape)
    else:
        images_paths = [path+"data32x32/" + f for f in listdir(path+"data32x32/") if ".jpg" in f]
        images_paths.sort()

        def load_img(image_path):
            im = imageio.imread(image_path)  # in RGB format
            return im

        data = [load_img(p) for p in images_paths]
        data = np.asarray(data)
        np.save(baked_path, data)
        print("data loaded as:", data.shape)

    data = np.clip((data + 0.5) / 256., 0., 1.)
    data = np.asarray(data, dtype=np.float32)

    # shuffle
    np.random.seed(42)
    np.random.shuffle(data)
    # randomize again
    t = 1000 * time.time()  # current time in milliseconds
    np.random.seed(int(t) % 2 ** 32)

    # train-test split
    split_idx = int(len(data) * train_test_split)
    train = data[0:split_idx]
    test = data[split_idx:]

    print("split with train=", train.shape, "test=", test.shape)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, **kwargs)

    input_size = int(32*32*3)
    return train_loader, test_loader, input_size


"""
cuda = False
batch_size = 20
train_loader, test_loader, input_size = load_celeb_a_32x32(cuda,batch_size=batch_size)
print("train_loader", len(train_loader)*batch_size, "test_loader", len(test_loader)*batch_size, "input_size=",input_size)
for sample in train_loader:
    t = sample.numpy()
    print("batch shape, min and max", t.shape, np.min(t.flatten()), np.max(t.flatten()))
    break
"""
