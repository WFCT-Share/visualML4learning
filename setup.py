from setuptools import setup, find_packages

setup(
    name='visualML4learning',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['pandas', 'matplotlib', 'numpy', 'joblib', 'easylogger4dev_alpha'],
    python_requires='>=3.10',
    author='DLMR-CODE',
    author_email='yuzhihao82@gmail.com',
    description='A description of your package',
    license='MIT',
    keywords='machine_learning, dev, easy, alpha'
)
