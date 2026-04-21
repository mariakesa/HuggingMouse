from setuptools import find_packages, setup

setup(
    name="HuggingMouse",
    version="0.3.1",
    description="Data analysis library for Allen Brain Observatory data",
    author="Maria Kesa",
    author_email="mariarosekesa@gmail.com",
    url="https://github.com/mariakesa/HuggingMouse",
    install_requires=[
        "allensdk @ git+https://github.com/AllenInstitute/AllenSDK.git",
        "scikit-learn>=1.2",
        "torch>=2.1",
        "pandas>=2.2,<3",
        "numpy>=1.26,<3",
        "transformers>=4.40,<5",
        "plotly>=5.9",
    ],
    extras_require={
        "agents": [
            "smolagents>=1.7.0",
        ],
        "dev": [
            "pytest>=8",
            "black>=24",
            "ruff>=0.5",
        ],
    },
    packages=find_packages("src"),
    package_dir={"": "src"},
)