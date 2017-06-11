import codecs


def bio_validator(file_path):
    for line in codecs.open(file_path, 'r', 'utf-8'):
        line = line.rstrip()
        if not line:
            continue
        if not all([item for item in line.split(' ')]):
            raise ValueError("bio file validation failed: %s" % line.split(' '))
