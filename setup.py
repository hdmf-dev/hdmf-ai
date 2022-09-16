# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import os
import shutil

with open('README.md', 'r') as fp:
    readme = fp.read()

pkgs = find_packages('src', exclude=['data'])
print('found these packages:', pkgs)

src_dir = os.path.dirname(__file__)
shutil.copytree(os.path.join(src_dir, 'ml'), os.path.join(src_dir, 'hdmf_ml', 'schema'), dirs_exist_ok=True)

reqs = [
    'hdmf',
    'scikit-learn'
]

setup_args = {
    'version': '0.0.1',
    'name': 'hdmf_ml',
    'description': 'A package for using the HDMF-ML schema',
    'long_description': readme,
    'long_description_content_type': 'text/x-rst; charset=UTF-8',
    'author': 'Andrew Tritt',
    'author_email': 'ajtritt@lbl.gov',
    'url': 'https://github.com/hdmf-dev/hdmf-ml',
    'license': "BSD",
    'install_requires': reqs,
    'packages': pkgs,
    'package_data': {'hdmf_ml': ["schema/*.yaml"]},
    'classifiers': [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    'keywords': 'python '
                'microbiome '
                'microbial-taxonomy '
                'cross-platform '
                'open-data '
                'data-format '
                'open-source '
                'open-science '
                'reproducible-research '
                'machine-learning',
    'zip_safe': False
}

if __name__ == '__main__':
    setup(**setup_args)
