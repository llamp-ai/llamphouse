from setuptools import setup, find_namespace_packages
from llamphouse import __version__

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name="llamphouse",
    version=__version__,
    author="llamp.ai",
    author_email="info@llamp.ai",
    description="LLAMP-House OpenAI Assistant Server",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=['llamphouse.core', 'llamphouse.core.*']),
    python_requires='>=3.10',
    install_requires=install_requires,
    package_data={},
)