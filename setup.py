from setuptools import setup, find_packages

setup(
    name='pdatakit',
    version='0.0.2',
    author='Hoang-Phat Vu',
    author_email='phatvucoder@gmail.com',
    description='A Python library for managing and processing datasets for machine learning workflows.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/phatvucoder/pdatakit',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'iterative-stratification',
        'pillow',
        'pyyaml',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    include_package_data=True,
)
