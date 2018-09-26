# Augment

A simple script for offline image augmentation using `multiprocessing` and
[PILlow](https://pillow.readthedocs.io).

Part of the interface is inspired by
[torchvision.transforms](http://pytorch.org/docs/torchvision/transforms.html).

## Usage

The script will load all the images in `src/**/*.jpg` and will save the
augmented images respecting the source directory hierarchy.

```python
from augment import Manager
import augment.transformations as T

transformations = [
    T.Scale(size=512, mutable=True),  # scales to 512px smalles side
    T.CycledColorCast(angle=90),  # walks the hue wheel in 90 degr. steps
    T.Blur(radius=2),  # Blur the Original&Scaled image, not the color cast
    T.ColorCast(angle=45, mutable=True),  # mutates the Original&Scaled img
    T.Blur(radius=3),  # Blurs the ColorCast img
    T.Distort(kind=T.BarrelDistortion(k=0.125)),
    T.Distort(kind=T.PincushionDistortion(k=0.125))
]

src = "data/src/"
dest = "data/dest/"

Manager(src=src,
        dest=dest,
        transformations=transformations,
        num_workers=4
        ).process()
```

## Installation:

```
git clone https://github.com/floringogianu/augment
cd augment

pip install -r requirements.txt
pip install -e .
```

## ToDo

- ~~Make it a `pip` plugin~~
- Proper handling of file extensions
- Better Exception Handling
- Add support for non-saving transformations
