# coding: utf-8

from . import config

class LiteralParameter():
    """
    A simple valued (string or numeric) parameter
    """

    def __init__(self, description):
        self.description = description

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.name)
        return value

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

class DatasourceParameter():
    """
    A file-based datasource declared in `config.ini`
    """

    def __init__(self, description):
        self.description = description

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.name)
        return DatasourceResolver(value)

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

class DatasourceResolver():
    """
    Resolves a DatasourceParameter according to current configuration.
    """

    def __init__(self, name):

        self.name = name

    def filename(self):
        """
        Resolves to this datasource filename
        """

        if self.name == 'off':
            return None

        return config.datasource(self.name).filename

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
        value = getattr(obj, self.name)
        return DatasetResolver(value)

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

class DatasetResolver():
    """
    Resolves a DatasetParameter according to current configuration.
    """

    def __init__(self, name):

        self.name = name

    def filename(self, mode='r', tileset='default', **kwargs):
        """
        Resolves to this datasource filename.
        If `tileset` is None, returns a global dataset.
        Otherwise, returns the single-file dataset for the specified tileset,
        such as a VRT file for tiled datasets.
        """

        if tileset is None:
            return config.filename(self.name, **kwargs)

        return config.tileset(tileset).filename(self.name, **kwargs)

    def tilename(self, mode='r', tileset='default', **kwargs):
        """
        Resolves to a specific tilename based on `kwargs` arguments
        """

        return config.tileset(tileset).tilename(self.name, **kwargs)

    def __repr__(self):
        return f'DatasetResolver({self.name})'

def Workflow():

    # def __init__(self, configuration=None):

    #     if configuration is None:
    #         self.config = config
    #     else:
    #         self.config = configuration

    def __init__(self):
        
        self.times = list()

    def __enter__(self):
        
        self.saved_workspace = config.workspace.copy()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        
        config.set_workspace(self.saved_workspace)
        self.saved_workspace = None

    def set_workdir(self, workdir):
        config.workspace._workdir = workdir

    def set_outputdir(self, outputdir):
        config.workspace._outputdir = outputdir

    def set_tileset(self, tileset):
        config.workspace._tileset = tileset

    def set_tiledir(self, tiledir):
        pass

    def record_execution_time(self, name, elapsed):

        self.times.append((name, elapsed))

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

