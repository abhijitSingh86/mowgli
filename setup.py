import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mowgli",
    version="0.0.1",
    author="Balu Chatbot",
    author_email="balu-chatbot@thoughtworks.com",
    description="Intent classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhijitSingh86/mowgli",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
