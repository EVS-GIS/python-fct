# coding: utf-8

import logging
import time
import click
# from ..cli.Decorators import pretty_time_delta
from . import config

def pretty_time_delta(delta):
    """
    See https://gist.github.com/thatalextaylor/7408395
    """

    days, seconds = divmod(int(delta), 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if days > 0:
        return '%d d %d h %d min %.0f s' % (days, hours, minutes, seconds)

    if hours > 0:
        return '%d h %d min %.0f s' % (hours, minutes, seconds)

    if minutes > 0:
        return '%d min %d s' % (minutes, seconds)

    return '%.1f s' % (delta,)

class LiteralParameter():
    """
    A simple valued (string or numeric) parameter
    """

    def __init__(self, description):

        self.description = description
        self.type = 'param'

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, obj, objtype=None):

        if obj is None:
            return self

        value = getattr(obj, self.name)
        return value

    def __set__(self, obj, value):

        if obj is None:
            return

        setattr(obj, self.name, value)

class DatasourceParameter():
    """
    A file-based datasource declared in `config.ini`
    """

    def __init__(self, description):

        self.description = description
        self.type = 'source'

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, obj, objtype=None):

        if obj is None:
            return self

        value = getattr(obj, self.name)
        return DatasourceResolver(value)

    def __set__(self, obj, value):

        if obj is None:
            return

        setattr(obj, self.name, value)

class DatasourceResolver():
    """
    Resolves a DatasourceParameter according to current configuration.
    """

    def __init__(self, value):

        self.key = value

    @property
    def none(self):
        """
        Return True if this parameter should resolve to None
        """

        return self.key is None or self.key == 'off'

    @property
    def name(self):
        """
        Return datasource's key
        """

        return self.key

    def filename(self):
        """
        Resolves to this datasource filename
        """

        if self.none:
            return None

        return config.datasource(self.key).filename

class DatasetParameter():
    """
    A file-based dataset declared in `datasets.yml`,
    either tiled or not
    """

    def __init__(self, description, type=None):

        self.description = description
        self.type = type

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, obj, objtype=None):

        if obj is None:
            return self

        value = getattr(obj, self.name)
        return DatasetResolver(value)

    def __set__(self, obj, value):

        if obj is None:
            return

        if isinstance(value, (str, dict)):

            setattr(obj, self.name, value)

        else:

            raise ValueError('Expected str or dict, got %s : %s' % (type(value), value))

class DatasetResolver():
    """
    Resolves a DatasetParameter according to current configuration.
    """

    def __init__(self, value):

        if isinstance(value, dict):

            self.key = value['key']
            self.tiled = value.get('tiled', True)
            self.args = {k: v for k, v in value.items() if k not in ('key', 'tiled')}

        else:

            self.key = value
            self.tiled = True
            self.args = None

    @property
    def none(self):
        """
        Return True if this parameter should resolve to None
        """

        return self.key is None or self.key == 'off'

    @property
    def name(self):
        """
        Return dataset's key
        """

        return self.key

    def arguments(self, kwargs=None):

        kwargs = kwargs if kwargs is not None else dict()

        if self.args is None:

            args = kwargs

        else:

            args = self.args.copy()
            args.update(kwargs)

        return args

    def filename(self, mode='r', tileset='default', **kwargs):
        """
        Resolves to this datasource filename.
        If `tileset` is None, returns a global dataset.
        Otherwise, returns the single-file dataset for the specified tileset,
        such as a VRT file for tiled datasets.
        """

        if self.none:
            return None

        args = self.arguments(kwargs)

        if not self.tiled or tileset is None:
            return config.filename(self.key, **args)

        return config.tileset(tileset).filename(self.key, **args)

    def tilename(self, mode='r', tileset='default', **kwargs):
        """
        Resolves to a specific tilename based on `kwargs` arguments
        """

        if self.none:
            return None

        args = self.arguments(kwargs)

        return config.tileset(tileset).tilename(self.key, **args)

    def __repr__(self):
        return f'DatasetResolver({self.key})'

class WorkflowContext():
    """
    A context manager that allows for specific execution configuration
    and records execution details.
    """

    # def __init__(self, configuration=None):

    #     if configuration is None:
    #         self.config = config
    #     else:
    #         self.config = configuration

    logger = logging.getLogger('workflow')

    def __init__(self, name='unnamed'):

        self.name = name
        self.times = list()

    def __enter__(self):

        self.saved_workspace = config.workspace.copy()
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        config.set_workspace(self.saved_workspace)
        self.saved_workspace = None

    def set_name(self, name):
        self.name = name

    def set_workdir(self, workdir):
        """
        Set current workspace's working directory
        """
        config.workspace._workdir = workdir

    def set_outputdir(self, outputdir):
        """
        Set output directory within workspace's working directory
        """
        config.workspace._outputdir = outputdir

    def set_tileset(self, tileset):
        """
        Set current default tileset
        """
        config.workspace._tileset = tileset

    def set_tiledir(self, tiledir):
        """
        Set default tileset's tile directory
        """
        pass

    def before_operation(self, operation, *args, **kwargs):
        """
        Before-execution hook
        """

        name = operation.__name__
        self.start_time = time.time()

    def after_operation(self, operation, *args, **kwargs):
        """
        After-execution hook
        """

        name = operation.__name__
        elapsed = time.time() - self.start_time
        click.echo(f'{name} : {pretty_time_delta(elapsed)}')

        self.record_execution_time(name, elapsed)

    def record_execution_time(self, operation, elapsed):
        """
        Record execution time for operation `name`
        """

        self.times.append((operation, elapsed))
        self.logger.info(
            'flow: %s operation: %s time: %s seconds: %f',
            self.name,
            operation,
            pretty_time_delta(elapsed),
            elapsed)

class FileResource:

    NETWORK_SUBDIR = 'NETWORK/METRICS'
    value = DatasetParameter('destination file (netcdf)', type='output')

    def __init__(self, key, axis=None):

        if axis is None:

            self.value = dict(
                key=key,
                tiled=False,
                subdir=self.NETWORK_SUBDIR)

        else:

            self.value = dict(
                key=key,
                tiled=False,
                axis=axis)

    def filename(self):

        return self.value.filename()

def test():

    class OperationParameter:
        """
        Example Operation input/output parameters definition
        """

        dem = DatasetParameter('input DEM')
        out = DatasetParameter('output raster')
        max_height = LiteralParameter('max height threshold')
        
        def __init__(self):
            """
            Default values
            """

            self.dem = 'dem'
            self.out = 'ax_nearest_height'
            self.max_height = 8.0

    params = OperationParameter()

    return params
