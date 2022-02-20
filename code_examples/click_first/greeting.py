"""create command line script, install it as executable code (hence why it is important to include meta info in
setup.py) you can they run you script by typing the specified console entry point in the command line"""
import click


@click.group()
def cli():
    pass


@cli.command()
@click.option('--count', default=1, help='# of greetings')
@click.argument('name')
def greet(count, name):
    for _ in range(count):
        click.echo(f'hello {name}!')


@cli.command()
def morning():
    click.echo('good morning...')
