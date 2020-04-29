import multiprocessing
from setuptools import setup, find_packages
import os
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "text_analysis_prep",
    version = "0.1",
    packages = find_packages(),

    # Dependencies on other packages:
    setup_requires   = [],
    install_requires = ['nltk>=3.5',
                        'PySimpleGUI>=4.18.2',
                        'vaderSentiment>=3.3.1'
                        ],

    #dependency_links = ['https://github.com/DmitryUlyanov/Multicore-TSNE/tarball/master#egg=package-1.0']
    # Unit tests; they are initiated via 'python setup.py test'
    test_suite       = 'nose.collector',
    #test_suite       = 'tests',
    tests_require    =['nose'],

    # metadata for upload to PyPI
    author = "Ken Flerlage",
    author_email = "paepcke@cs.stanford.edu",
    description = "Creating stemmed text statistics",
    long_description_content_type = "text/markdown",
    long_description = long_description,
    license = "BSD",
    keywords = "text analysis",
    url = " https://www.flerlagetwins.com/2019/09/text-analysis.html",   # project home page, if any
)

# Ensure that the nltk data sets are present:
HOME = os.environ.get('HOME')
STANDARD_LOC = os.path.join(HOME, 'nltk_data')
if not os.path.exists(STANDARD_LOC):
    import nltk
    nltk.downloader.download(
            info_or_id='all',
            download_dir=STANDARD_LOC)
                          

