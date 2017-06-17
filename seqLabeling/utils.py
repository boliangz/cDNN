import codecs


def bio_validator(file_path):
    for line in codecs.open(file_path, 'r', 'utf-8'):
        line = line.rstrip()
        if not line:
            continue
        if not all([item for item in line.split(' ')]):
            raise ValueError("bio file validation failed: %s" % line.split(' '))


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()
