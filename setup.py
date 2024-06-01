from setuptools import find_packages, setup

setup(name='HuggingMouse',
      version='0.2.0',
      description='Data analysis library for Allen Brain Observatory data',
      author='Maria Kesa',
      author_email='mariarosekesa@gmail.com',
      url='https://github.com/mariakesa/HuggingMouse',
      install_requires=[
          'scikit-learn==1.2.2',
          'torch==1.13.1',
          'pandas==1.5.3',
          'numpy==1.23.5',
          'transformers==4.31.0',
          'allensdk==2.16.2',
          'plotly==5.9.0'
      ],
      packages=find_packages("src"),
      package_dir={"": "src"}
      )
