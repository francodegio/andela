from setuptools import setup, find_packages

setup(
    name='andela',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'fastapi',
        'pandas',
        'scikit-learn',
        'catboost'
    ]
)