# coding: utf-8

from . import config

class LiteralParameter():

    def __init__(self, description):
        self.description = description

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.name)
        return value

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

class DatasetParameter():

    def __init__(self, description, iotype=None):
        self.description = description
        self.iotype = iotype

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.name)
        return DatasetResolver(value)

    def __set__(self, obj, value):
        setattr(obj, self.name, value)

class DatasetResolver():
    """
    mode:

        r: read-only, check file exists
        r?: read-only, file may not exist
        w: write, update existing file or create new file
        rw:
    """

    def __init__(self, name):

        self.name = name

    def filename(self, mode='r', tileset='default', **kwargs):

        if tileset is None:
            return config.filename(self.name, **kwargs)

        return config.tileset(tileset).filename(self.name, **kwargs)

    def tilename(self, mode='r', tileset='default', **kwargs):

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

