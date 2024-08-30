
from setuptools import setup, find_packages

setup(
    name="BinWizard",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
    ],
    package_data={
        'BinWizard': ['continuous_target.py', 'binary_target.py'],
    },
    entry_points={
        'console_scripts': [
            'binGUI=BinWizard.bin_target_launcher:binGUI',
            'contGUI=BinWizard.cont_target_launcher:contGUI',
        ],
    },
    author="Iván Pastor",
    author_email="ivanpastorsanz@gmail.com",
    description="Paquete para optimizar binnings de variables continuas con target dicotómico o continuo",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ivanpast/BinWizard",
)
