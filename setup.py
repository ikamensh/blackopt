from setuptools import setup

version = 0.8


def get_requirements():
    with open("requirements.txt") as fp:
        return fp.read()


setup(
    name="blackopt",
    packages=[
        "blackopt",
        "blackopt.abc",
        "blackopt.algorithms",
        "blackopt.examples",
        "blackopt.examples.launchers",
        "blackopt.examples.problems",
        "blackopt.examples.problems.tsp",
        "blackopt.util",
    ],
    # package_dir={"": "."},
    version=version,
    description="black box optimization",
    author="Ilya Kamenshchikov",
    keywords=["optimization"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    license="MIT",
    long_description="In a far, far away galaxy...",
    install_requires=get_requirements(),
    python_requires=">=3.7",
    url="https://github.com/ikamensh/optim",
)
