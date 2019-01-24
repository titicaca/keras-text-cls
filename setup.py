from __future__ import print_function
from setuptools import setup, find_packages
import sys
import io

setup(
    name="keras_text_cls",
    version="0.1.0",
    author="Titicaca",
    author_email="lake_titicaca@outlook.com",
    description="Text Classification Lib on Keras",
    long_description=io.open("README.md", encoding="UTF-8").read(),
    license="MIT",
    packages=find_packages(),
    entry_points={
    },
    data_files=[('data', []),
                ('doc', ['README.md']),
                ('conf', ['keras_text_cls/conf/keras_text_cls.cfg']),
                ('dict', [])],
    include_package_data=True,
    classifiers=[
        "Environment :: Web Environment",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: Chinese',
        'Operating System :: MacOS',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Topic :: NLP',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
            'jieba>=0.39',
            'gensim>=3.4.0',
            'tensorflow>=1.12.0',
            'pytest>=3.6.3',
            'pandas>=0.23.3',
            'numpy>=1.14.3',
            'Cython>=0.28.5',
            'fasttext>=0.8.3',
            'keras>=2.2.4'
        ],
    zip_safe=True,
)
