import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple_pycuda",
    version="0.0.1",
    author="Igor Machado and Rodolfo Araujo",
    author_email="igor.machado@ime.uerj.br",
    description="A simple wrapper for CUDA functions in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igormcoelho/simple-pycuda",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: MIT License",
    ],
#    data_files=[
#    	('cuda', ['simple_pycuda/cudapp.cu', 'simple_pycuda/cudapp.h']),
#    ],
)
