# coding: utf-8

"""
Command Line Interface

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from .Decorators import (
    fct_entry_point,
    parallel,
    aggregate,
    starcall,
    command_info,
    pretty_time_delta
)

from .Options import (
    overwritable,
    verbosable,
    parallel_opt
)
