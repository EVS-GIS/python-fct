from qgis.core import *
from qgis.analysis import QgsNativeAlgorithms
from processing import Processing
from fct.cli.helpers import start_app, execute_algorithm
from fct.FluvialCorridorToolbox import PROVIDERS
import os

# Bootstrap code

os.environ['QGIS_PREFIX_PATH'] = '/usr'
# QgsApplication.setPrefixPath('/usr')
# QgsApplication.setPluginPath('/usr/lib/qgis/plugins')
# QgsApplication.setPkgDataPath('/usr/share/qgis')
# QgsApplication.addLibraryPath('/usr/lib/qgis/plugins')
# QgsApplication.addLibraryPath('/usr/bin')
# QgsApplication.addLibraryPath('/usr/lib/x86_64-linux-gnu/qt5/plugins')
QgsApplication.setApplicationName('QGIS3')
QgsApplication.setOrganizationName('QGIS')
QgsApplication.setOrganizationDomain('qgis.org')

app = start_app(False, True)
registry = app.processingRegistry()
Processing.initialize()

providers = [c() for c in PROVIDERS]
providers.append(QgsNativeAlgorithms())
for provider in providers:
    registry.addProvider(provider)

print(app.showSettings())

# pr = QgsProviderRegistry.instance()
# for provider in pr.providerList():
#     print(provider)

# Load models from repository ...

# Info

def isRequired(parameter):
    """
    Return False if parameter `parameter` is optional.
    """
    return int(parameter.flags() & QgsProcessingParameterDefinition.FlagOptional) == 0

def algorithmHelp(algorithm):

    for param in algorithm.parameterDefinitions():
        required = '[REQ]' if isRequired(param) else ''
        print(param.name(), required, param.defaultValue(), param.description())

# Execute wrapper

def execute(algorithm_id, **parameters):

    params = {key.upper(): parameters[key] for key in parameters}
    return execute_algorithm(algorithm_id, **params)