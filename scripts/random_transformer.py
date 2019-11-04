from common import *

random_transformer = transforms.Compose([
                       transforms.ToPILImage(),
                       transforms.ColorJitter(brightness=[0.6, 1.2], contrast=0.0, saturation=0.0, hue=0.0),
                       transforms.Resize(IMAGE_SIZE),
                       transforms.RandomCrop(IMAGE_SIZE, pad_if_needed=True, fill=0, padding_mode='constant'),
                       transforms.RandomAffine(0, translate=(0.05, 0.05), scale=None, shear=None, resample=False, fillcolor=0),
                       transforms.ToTensor(),
                       torchvision.transforms.RandomErasing(p=0.5, scale=(0.5, 0.5), ratio=(0.5, 0.5), value=0, inplace=False)])
