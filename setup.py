from setuptools import setup

requirements = [
    'PyMatching==0.2.4',
    'tqdm>=4.59.0',
]

setup(
    name='bn3d',
    description='Biased Noise 3D Codes',
    url='http://github.com/ehua7365/bn3d',
    author='Eric Huang',
    email='ehuang1@perimeterinstitute.ca',
    packages=['bn3d'],
    zip_safe=False,
    requirements=requirements
)
