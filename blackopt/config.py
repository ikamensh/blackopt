import os

default_workspace = "_blackopt_workspace"
_rootdir = default_workspace

def set_rootdir(path):
    path = os.path.expanduser(path)
    global _rootdir
    _rootdir = path

def prepend_rootdir(prefix):
    prefix = os.path.expanduser(prefix)
    path = os.path.join(prefix, default_workspace)
    global _rootdir
    _rootdir = path

def get_rootdir():
    return _rootdir