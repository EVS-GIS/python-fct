# coding: utf-8

import os
from configparser import ConfigParser
from collections import namedtuple
import click
import fiona

Tile = namedtuple('Tile', ('gid', 'row', 'col', 'x0', 'y0'))

__options__ = None
__tileindex__ = None

def config(configfile=None):
    """
    Read config file, defaults to 'config.cfg'
    """

    global __options__

    if __options__ is not None:
        return __options__

    if not configfile:
        configfile = os.path.join(os.path.dirname(__file__), 'config.ini')

    click.secho('Config from %s' % configfile, fg='yellow')

    parser = ConfigParser()
    parser.read(configfile)
    options = dict()

    for section in parser.sections():
        for key, value in parser.items(section):
            options['%s.%s' % (section.lower(), key)] = value

    __options__ = options

    return options

def parameter(key):
    """
    Configuration defined parameter
    """
    options = config()
    return options.get(key)

def tileindex():
    """
    Populate tile index from shapefile
    """

    global __tileindex__

    if __tileindex__ is not None:
        return __tileindex__

    options = config()
    shapefile = options['workspace.tileindex']
    index = dict()

    with fiona.open(shapefile) as fs:
        for feature in fs:
            props = feature['properties']
            values = [props[k] for k in ('GID', 'ROW', 'COL', 'X0', 'Y0')]
            tile = Tile(*values)
            index[(tile.row, tile.col)] = tile

    __tileindex__ = index

    return index

def basedir():

    options = config()
    return options['workspace.basedir']

def workdir():

    options = config()
    return options['workspace.workdir']

def filename(key, kind='output', **kw):

    options = config()
    pattern = options['%s.%s' % (kind, key)]
    return os.path.join(workdir(), pattern % kw)

def fileset(keys, kind='output', **kw):

    return {key: filename(key, kind, **kw) for key in keys}
