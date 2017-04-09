import time
from augment import Manager
from augment import transformations as T


if __name__ == "__main__":
    """ Augment usage.

        Args:
        - `src`: root of the directory containing folders for each of the
          classes in your dataset.
        - `dest`: destination folder for the augmented imaes. Needs to contain
          same class folders as `src`.
        - `num_workers`: no of processes to be used for augmenting the data.
    """

    src = "data/src/"
    dest = "data/dest/"

    start = time.time()

    transformations = [
        T.Scale(size=512, mutable=True),  # scales to 512px smalles side
        T.CycledColorCast(angle=90),  # walks the hue wheel in 90 degr. steps
        T.Blur(radius=2),  # Blur the Original&Scaled image, not the color cast
        T.ColorCast(angle=45, mutable=True),  # mutates the Original&Scaled img
        T.Blur(radius=3),  # Blurs the ColorCast img
        T.Distort(kind=T.BarrelDistortion(k=0.125)),
        T.Distort(kind=T.PincushionDistortion(k=0.125))
    ]

    Manager(src=src,
            dest=dest,
            transformations=transformations,
            num_workers=4
            ).process()

    time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print("Time: ", time_str)
