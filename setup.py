# setup.py

from setuptools import setup, find_packages

setup(
    name="easy_functions",
    version="0.1.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="A Python library with 50+ utility functions.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YourGitHubUsername/easy_py",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
