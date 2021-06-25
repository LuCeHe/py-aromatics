import os

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', 'data',))

if not os.path.isdir(DATADIR):
    os.mkdir(DATADIR)