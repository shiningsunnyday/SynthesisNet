import logging

def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

