from setuptools import setup, find_packages

setup(
    name='TiML',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy==1.25.2',
    ],
    author='Tim Koornstra',
    description='An educational machine learning module.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/TimKoornstra/TiML',
)
