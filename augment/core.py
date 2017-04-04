""" Augment core.

    Manager and Worker classes.
"""
import os
import multiprocessing as mp
from PIL import Image


class Manager(object):
    def __init__(self, src, dest, transformations, num_workers=4):
        self.src = src
        self.dest = dest
        self.num_workers = num_workers

        self.queue = mp.Queue()
        self.workers = [Worker(w, src, dest, self.queue, transformations)
                        for w in range(num_workers)]

    def process(self):
        self._fill_queue()
        for w in self.workers:
            w.start()
        for w in self.workers:
            w.join()

    def _fill_queue(self):
        for (dirpath, dirnames, filenames) in os.walk(self.src):
            for f in filenames:
                if f is not ".DS_Store":
                    self.queue.put(dirpath + "/" + f)


class Worker(mp.Process):
    def __init__(self, pidx, src, dest, queue, transformations):
        mp.Process.__init__(self)

        self.pidx = pidx
        self.src = src
        self.dest = dest
        self.queue = queue
        self.transformations = transformations

    def run(self):
        """ Main worker loop. """

        print("[%d] Starting processing." % (self.pidx))
        img_no = 0

        while not self.queue.empty():
            src_path = self.queue.get()
            try:
                src_img = Image.open(src_path)
            except IOError as e:
                print("Cannot open: ", src_path)
                print("Error: ", e)

            prev_info = ""
            for t in self.transformations:
                dest_img = t.transform(src_img)
                info = prev_info + t.get_info()

                if isinstance(dest_img, list):
                    for img_tuple in dest_img:
                        img, info = img_tuple
                        self._save(img, src_path, prev_info + info)
                else:
                    self._save(dest_img, src_path, info)

                    # keep this image for further processing
                    if t.mutable:
                        src_img = dest_img
                        prev_info += t.get_info()
            img_no += 1

        print("[%d] Finished processing %03d images." % (self.pidx, img_no))

    def _save(self, img, src_path, info=None):
        dest_path = self._get_dest_path(src_path, info)
        # print(dest_path)
        img.save(dest_path)

    def _get_dest_path(self, src_path, info):
        src_path, src_fn = os.path.split(src_path)
        dest_path = src_path.replace(self.src, self.dest)
        dest_fn = src_fn.replace(".jpg", (str(info) if info else "") + ".jpg")
        dest_path += ("/" + dest_fn)
        return dest_path
