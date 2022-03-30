from setuptools import setup, find_packages

requirements = [
    'PyMatching',
    'tqdm',
    'python-dotenv',
    'qecsim',
    'pandas',
    'flask'
]

setup(
    name='panqec',
    description='Simulation and visualization of quantum error correcting codes',
    url='http://github.com/ehua7365/panqec',
    author='Eric Huang',
    email='ehuang1@perimeterinstitute.ca',
    packages=find_packages(),
    zip_safe=False,
    requirements=requirements,
    scripts=['bin/panqec'],
)