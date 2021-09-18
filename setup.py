from setuptools import setup, find_packages

requirements = [
    'PyMatching',
    'tqdm',
    'python-dotenv',
    'qecsim',
    'pandas',
]

setup(
    name='bn3d',
    description='Biased Noise 3D Codes',
    url='http://github.com/ehua7365/bn3d',
    author='Eric Huang',
    email='ehuang1@perimeterinstitute.ca',
    packages=find_packages(),
    zip_safe=False,
    requirements=requirements,
    scripts=['bin/bn3d'],
)
