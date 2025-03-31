from setuptools import setup, find_packages

setup(
    name="actividad",
    version="0.0.1",
    author="Elizabeth Cristina Guerra",
    author_email="elizabeth.guerra@est.iudigital.edu.co",
    description="Se desarrolla actividad numero 3",
    py_modules=["data.ipynb"],
    install_requires=[
        "kagglehub[pandas-datasets]>=0.3.8",
        "pandas",
        "requests",
        "matplotlib"
    ]
    
    
)
