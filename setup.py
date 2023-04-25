from setuptools import find_packages, setup

setup(
    name="gpsd",
    packages=find_packages(),
    version="0.1.0",
    description="Gaussian Process Self Distillation",
    author="Kenneth Borup",
    author_email="kennethborup@math.au.dk",
    license="MIT",
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "scipy",
    ],
)
