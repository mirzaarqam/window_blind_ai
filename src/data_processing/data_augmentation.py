import albumentations as A

def get_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(p=0.3),
        A.RandomShadow(p=0.1),
        A.CLAHE(p=0.1),
    ])