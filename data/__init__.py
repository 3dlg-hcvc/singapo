datamodules = {}

def register(name):
    def decorator(cls):
        datamodules[name] = cls
        return cls
    return decorator

def make(name, config):
    dm = datamodules[name](config)
    return dm

from . import datamodule