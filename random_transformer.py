from common import *

random_transformer = transforms.Compose([
                       transforms.ToPILImage(),
                       transforms.ColorJitter(brightness=[0.8, 1.2], contrast=0.0, saturation=0.0, hue=0.0),
                       transforms.Resize(image_size),
                       transforms.RandomCrop(image_size, pad_if_needed=True, fill=0, padding_mode='constant'),
                       transforms.ToTensor()])
