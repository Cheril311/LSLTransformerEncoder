from setuptools import setup, find_packages

setup(
    name='LSLEncoder',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'fairseq>=0.10.0',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'pretrain=scripts.pretrain:main',
            'posttrain=scripts.posttrain:main',
        ],
    },
)
