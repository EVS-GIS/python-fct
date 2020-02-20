import click

def success(msg):
    click.secho(msg, fg='green')

def info(msg):
    click.secho(msg, fg='cyan')

def warning(msg):
    click.secho(msg, fg='red')

def important(msg):
    click.secho(msg, fg='yellow')

