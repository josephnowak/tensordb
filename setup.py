from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='TensorDB',
    version='0.25',
    description='Database based in a file system storage combined with Xarray and Zarr',
    author='Joseph Nowak',
    author_email='josephgonowak97@gmail.com',
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: General',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='Database Files Xarray Handler Zarr Store Read Write Append Update Upsert Backup Delete S3',
    packages=find_packages(),
    install_requires=required
)
