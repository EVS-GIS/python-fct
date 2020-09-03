# coding: utf-8

"""
Common Click Options

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import click

overwritable = click.option(
    '--overwrite', '-w',
    default=False,
    help='Overwrite existing output ?',
    is_flag=True)

verbosable = click.option(
    '--verbose', '-v',
    default=False,
    help='Print verbose messages ?',
    is_flag=True)

quiet_opt = click.option(
    '--quiet/--no-quiet', '-q',
    default=False,
    help='Suppress message output ?')

parallel_opt = click.option(
    '--processes', '-j',
    default=1,
    help="Execute j parallel processes")
