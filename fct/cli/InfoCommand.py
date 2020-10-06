# coding: utf-8

"""
Fluvial Corridor Toolbox

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from collections import defaultdict
from textwrap import indent
import click

from ..config import config
from .. import __version__ as version


@click.group()
def cli():
    """
    Fluvial Corridor Toolbox
    """
    pass

@cli.command('version')
def print_version():
    """
    Print version and exit
    """

    click.echo(version)

@cli.command()
def citation():
    """
    Print citation reference
    """

    click.secho('Fluvial Corridor Toolbox', fg='green')
    click.secho('version %s' % version)
    click.secho('description ...')
    click.secho('cite me ...')
    click.secho('GitHub Link ...')

@cli.command()
@click.argument('name')
@click.option('--filename', '-f', default=False, is_flag=True, help='Search by filename')
@click.pass_context
def describe(ctx, name, filename):
    """
    Describe dataset
    """

    # pylint:disable=protected-access

    config.default()

    if filename:

        reverse = defaultdict(list)

        for dataset in config._workspace._datasets.values():
            if dataset._filename is not None:
                reverse[dataset._filename].append(dataset)

        if name in reverse:

            count = len(reverse[name])
            if count > 1:
                click.secho('-- Found %d datasets' % count, fg='yellow')

            for dataset in reverse[name]:

                click.echo()
                print_dataset_info(dataset)

        else:

            click.echo('No such dataset with filename %s' % name)
            ctx.exit(1)

    else:

        try:

            dataset = config.dataset(name)
            print_dataset_info(dataset)

        except KeyError:

            click.echo('No such dataset %s' % name)
            ctx.exit(1)

def print_dataset_info(dataset):
    """
    Print dataset info to console
    """

    click.echo('%13s: %s' % ('name', dataset.name))
    
    click.secho(
        '%13s: %s' % (
            'description',
            indent(dataset.properties['description'], 15*' ').strip()),
        fg='cyan')

    click.echo('%13s: %s' % ('subdir', dataset._subdir))

    if dataset._filename is None:
        click.secho('%13s: %s' % ('filename', 'Has only tiles'), fg='yellow')
    else:
        click.echo('%13s: %s' % ('filename', dataset._filename))

    if dataset._tilename is None:
        click.secho('%13s: %s' % ('tile template', 'No tiles'), fg='yellow')

    else:
        click.echo('%13s: %s' % ('tile template', dataset._tilename))

    for key, value in dataset.properties.items():
        if key not in ('description', 'filename', 'tiles', 'subdir'):
            click.echo('%13s: %s' % (key, value))
