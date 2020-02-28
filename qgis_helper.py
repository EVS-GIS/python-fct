# -*- coding: utf-8 -*-

"""
Start QGis Application Helper
Adapted from qgis.testing

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
import sys
import platform
import atexit
import click

from qgis.core import ( # pylint: disable=import-error,no-name-in-module
    QgsApplication,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterDefinition
)

from qgis.analysis import ( # pylint: disable=import-error,no-name-in-module
    QgsNativeAlgorithms
)

from processing import Processing # pylint: disable=import-error,no-name-in-module

from processing.tools.dataobjects import ( # pylint: disable=import-error,no-name-in-module
    createContext
)

from fct.FluvialCorridorToolbox import PROVIDERS # pylint: disable=import-error,no-name-in-module

def start_app(gui=True, cleanup=True):
    """
    Will start a QgsApplication and call all initialization code like
    registering the providers and other infrastructure.
    It will not load any plugins.
    You can always get the reference to a running app by calling `QgsApplication.instance()`.
    The initialization will only happen once, so it is safe to call this method repeatedly.

    Parameters
    ----------
    cleanup: Do cleanup on exit. Defaults to true.

    Returns
    -------
    QgsApplication
    A QgsApplication singleton
    """

    global QGISAPP # pylint: disable=global-variable-undefined

    try:
        QGISAPP
    except NameError:

        # In python3 we need to convert to a bytes object (or should
        # QgsApplication accept a QString instead of const char* ?)
        try:
            argvb = [os.fsencode(arg) for arg in sys.argv]
        except AttributeError:
            argvb = ['']

        if platform.system() == 'Darwin':
            QgsApplication.addLibraryPath(os.path.expandvars('$QGIS_PREFIX_PATH/../Plugins'))
            QgsApplication.addLibraryPath(os.path.expandvars('$QGIS_PREFIX_PATH/../Plugins/qgis'))
            QgsApplication.addLibraryPath(os.path.expandvars('$QGIS_PREFIX_PATH/../MacOS'))

        # Note: QGIS_PREFIX_PATH is evaluated in QgsApplication -
        # no need to mess with it here.
        QGISAPP = QgsApplication(argvb, gui)

        QGISAPP.initQgis()
        # click.echo(QGISAPP.showSettings())
        Processing.initialize()

        def debug_log_message(message, tag, level):
            """ Print debug message on console """
            click.secho('{}({}): {}'.format(tag, level, message), fg='yellow')

        QgsApplication.instance().messageLog().messageReceived.connect(debug_log_message)

        if cleanup:

            @atexit.register
            def exitQgis(): # pylint: disable=unused-variable
                """ Exit Qgis Application """
                QGISAPP.exitQgis()

    return QGISAPP

def execute_algorithm(algorithm_id, feedback=None, **parameters):

    algorithm = QgsApplication.processingRegistry().createAlgorithmById(algorithm_id)

    if feedback is None:
        feedback = QgsProcessingFeedback()

    context = createContext(feedback)

    parameters_ok, msg = algorithm.checkParameterValues(parameters, context)
    if not parameters_ok:
        raise QgsProcessingException(msg)

    if not algorithm.validateInputCrs(parameters, context):
        feedback.reportError(
            Processing.tr('Warning: Not all input layers use the same CRS.\nThis can cause unexpected results.'))

    results, execution_ok = algorithm.run(parameters, context, feedback)

    if execution_ok:
        return results
    else:
        msg = Processing.tr("There were errors executing the algorithm.")
        raise QgsProcessingException(msg)

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