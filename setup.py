
from setuptools import setup, find_packages

setup(
    name="BinWizard",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",  # Incluye otras dependencias necesarias
    ],
    entry_points={
        'console_scripts': [
            'bin-binary-target=scripts.run_bin_binary:main',
            'bin-continuous-target=scripts.run_bin_continuous:main',
        ],
    },
    author="Iván Pastor",
    author_email="ivanpastorsanz@gmail.com",
    description="Paquete para optimizar binnings de variables continuas con target dicotómico o continuo",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ivanpast/BinWizard",  # Cambia esto si tienes un repositorio
)
