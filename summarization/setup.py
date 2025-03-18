from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="eu-legal-summarization",
    version="0.1.0",
    author="Alexander Benady",
    author_email="",  # Add your email if desired
    description="Summarization module for EU legal documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",  # Add repository URL if available
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "run-summarization=summarization.src.run_pipeline:main",
        ],
    },
)
