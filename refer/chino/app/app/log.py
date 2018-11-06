import sys
import os
import logging


def get_logger():
    logger = logging.getLogger('nikkei_app')
    logger.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    log_file_path = os.path.join('/', 'tmp', 'nikkei_app', str(os.getpid()), 'main.log')

    if not os.path.isdir(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)  # log all

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.debug("Logged to %(log_file_path)s" % locals())

    return logger
