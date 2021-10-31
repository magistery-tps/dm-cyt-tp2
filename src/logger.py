import logging
import sys

def setup_logger(
    format   = '%(asctime)s - %(name)s - %(levelname)s: %(message)s', 
    level    = logging.INFO,
    filename = None,
    stream   = sys.stdout
):
    logging.basicConfig(format=format, level=level, stream=sys.stdout)