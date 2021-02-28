from setuptools import find_packages, setup


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='kaggle_titanic',
    author='yota-p',
    license='MIT',
    install_requires=_requires_from_file('requirements.txt'),
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"]
)
