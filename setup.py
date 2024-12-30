from setuptools import setup, find_namespace_packages

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

def get_version():
    with open("llamphouse/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

setup(
    name="llamphouse",
    version=get_version(),
    author="llamp.ai",
    author_email="info@llamp.ai",
    description="LLAMPHouse OpenAI Assistant Server",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=['llamphouse.core', 'llamphouse.core.*']),
    python_requires='>=3.10',
    install_requires=install_requires,
    package_data={},
)