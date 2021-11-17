import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='GenericTools',
    version='0.0.1',
    author='Luca Herrtti',
    author_email='luca.herrtti@gmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lucehe/GenericTools',
    project_urls = {
        "Bug Tracker": "https://github.com/mike-huls/GenericTools/issues"
    },
    license='MIT',
    packages=['GenericTools'],
    install_requires=['tensorflow', 'sacred', 'matplotlib'],
)