from setuptools import setup, find_packages

with open('requirements/common.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='panqec',
    version='0.1.6',
    license='MIT',
    description='Simulation and visualization of quantum error correcting codes',
    url='http://github.com/panqec/panqec',
    author='Eric Huang',
    email='ehuang1@perimeterinstitute.ca',
    packages=find_packages(),
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    scripts=['bin/panqec'],
)
