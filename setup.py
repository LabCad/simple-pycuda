import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "simplepycuda",
    version = "0.0.2.4",
    author = "Igor Machado Coelho and Rodolfo Pereira Araujo",
    author_email = "igor.machado@ime.uerj.br",
    description="A simple wrapper for CUDA functions in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igormcoelho/simple-pycuda",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 2',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: C++'
    ],
)
