import os
import re
from setuptools import find_packages, setup

def get_requires():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


extra_require = {
    "deepspeed": ["deepspeed"],
}


def main():
    setup(
        name="toyreclm",
        version="1.0.0",
        author="Yu-Ran Gu",
        author_email="guyr" "@" "nju.lamda.edu.cn",
        description="A toy large model for recommender system",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["LLaMA2", "SASRec", "large model", "recommender system", "Actions speak louder than words"],
        license="MIT License",
        url="https://github.com/glb400/Toy-RecLM",
        package_dir={"": "./"},
        packages=find_packages("./"),
        python_requires=">=3.8.0",
        install_requires=get_requires(),
        extras_require=extra_require,
        classifiers=[
        ],
    )


if __name__ == "__main__":
    main()