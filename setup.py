from setuptools import setup, find_packages

setup(
    name="tokenframe",
    version="0.1.0",
    description="LLM API cost optimization framework",
    author="Faustas",
    python_requires=">=3.10",
    packages=find_packages(
        exclude=[
            "tests",
            "tests.*"]),
    include_package_data=True,
    package_data={
        "tokenframe": ["economics/pricing.json"],
        "benchmarks": ["studybuddy/fixtures/questions.json"],
    },
    install_requires=[
        "anthropic>=0.25.0",
        "python-dotenv",
        "sentence-transformers>=2.2.0",
        "matplotlib>=3.5.0",
    ],
    entry_points={
        "console_scripts": [
            "tokenframe=tokenframe.cli:main",
        ],
    },
)
