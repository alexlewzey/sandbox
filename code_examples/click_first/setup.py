from setuptools import setup

setup(
    name='greeting',
    version='0.1',
    py_modules=['greeting'],
    install_requires=[
        'click'
    ],
    entry_points="""
        [console_scripts]
        greeting=greeting:cli
    """,
)