from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='topic-modelling-algorithms',
    version='0.0.0',
    description='A small collection of topic modelling algorithms',
    long_description=readme,
    author='Angie Pinchbeck',
    author_email='angie.pinchbeck@gmail.com',
    url='https://github.com/apinchbeck/topic-modelling-algorithms',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'pandas',
    ]
)
