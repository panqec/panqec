from setuptools import setup, find_packages

requirements = ["PyMatching==0.2.4",
                "tqdm==4.59.0",
                "python-dotenv==0.17.0",
                "psutil==5.8.0",
                "ldpc==0.1.0",
                "click==8.1.2"]

setup(
    name='panqec',
    version='0.1.2',
    license='MIT',
    description='Simulation and visualization of quantum error correcting codes',
    url='http://github.com/panqec/panqec',
    author='Eric Huang',
    email='ehuang1@perimeterinstitute.ca',
    packages=find_packages(),
    zip_safe=False,
    install_requires=requirements,
    scripts=['bin/panqec'],
)