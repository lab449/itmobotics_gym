import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='itmobotics_gym',
    version='0.0.2',
    author='TexnoMann',
    author_email='texnoman@itmo.ru',
    description='Package with gym enviroments from ITMO Robotics Lab',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ITMORobotics/itmobotics_gym",
    project_urls={
        "Bug Tracker": "https://github.com/ITMORobotics/itmobotics_gym/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT',
    download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',
    include_package_data=True,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={"": ["README.md", "LICENSE.txt", "*.json", "src/itmobotics_gym/envs.json"]},
    install_requires=[
        "numpy >=1.20.0",
        "gym",
        "jsonschema",
        "itmobotics-sim"
   ]
)
