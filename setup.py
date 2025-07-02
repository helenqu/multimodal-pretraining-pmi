from setuptools import setup, find_packages

setup(
    name="clip_pretraining",
    version="0.1",
    package_dir={"": "src"},  # Indicate that packages are in the src directory
    packages=find_packages(where="src"),  # Find packages within src
    install_requires=[
        # Dependencies
    ],
)
