"""Setup configuration for Principal Data Science Decision Agent."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="principal-ds-agent",
    version="0.1.0",
    author="Principal Data Science Team",
    author_email="ds-team@example.com",
    description="A comprehensive Principal Data Science Decision Agent for financial services",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sushil-tata/claude",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "ipython>=8.14.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ds-agent=agent.decision_agent:main",
        ],
    },
)
