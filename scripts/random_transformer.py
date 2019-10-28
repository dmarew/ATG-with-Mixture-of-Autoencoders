from common import *

random_transformer = transforms.Compose([
                       transforms.ToPILImage(),
                       transforms.ColorJitter(brightness=[0.6, 1.2], contrast=0.0, saturation=0.0, hue=0.0),
                       transforms.Resize(IMAGE_SIZE),
                       transforms.RandomCrop(IMAGE_SIZE, pad_if_needed=True, fill=0, padding_mode='constant'),
                       transforms.ToTensor()])
