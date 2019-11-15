from common import *

random_transformer = transforms.Compose([
                       transforms.ToPILImage(),
                       transforms.ColorJitter(brightness=[0.6, 1.4], contrast=0.0, saturation=0.0, hue=(-0.02, 0.02)),
                       transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                       #transforms.RandomCrop(IMAGE_SIZE, pad_if_needed=True, fill=0, padding_mode='constant'),
                       transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.90, 1.1), shear=15, resample=False, fillcolor=0),
                       transforms.ToTensor(),
                       torchvision.transforms.RandomErasing(p=0.7, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(0.5, 0.5, 0.5), inplace=False)])
pil_to_tensor = transforms.Compose([transforms.ToTensor()])
