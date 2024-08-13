from setuptools import setup, find_packages

setup(
    name="multilingual-transformer",
    version="0.1.0",
    description="A multilingual transformer model with language-specific layers",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Cheril311/LSLTransformerEncoder/",  # Replace with your repo URL
    packages=find_packages(include=['models', 'scripts', 'utils']),
    install_requires=[
        'torch>=1.9.0',  # Specify the PyTorch version required
        'fairseq>=0.10.2',  # Specify the fairseq version required
        'numpy>=1.21.0',
        # Add any other dependencies your project needs
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Specify the Python version required
    include_package_data=True,
)

