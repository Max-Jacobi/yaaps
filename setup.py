from setuptools import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# import numpy


setup(
    name="yaaps",
    version="0.0.1",
    author="Maximilian Jacobi",
    author_email="maximilian.jacobi@uni-jena.de",
    packages=["yaaps"],
    license='MIT',
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "matplotlib",
        "tqdm",
        "simtroller"
   ],
    python_requires='>=3.7',
    description="Routines to post-process and plot output from GRAthena++",
)
