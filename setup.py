import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Armadillo (C++ linear algebra) is needed only for the optional C++ training
# backend (cool_ext_arma). Override these if it lives elsewhere, e.g. Intel
# Homebrew uses /usr/local:  ARMADILLO_INCLUDE=/usr/local/include pip install -e .
ARMA_INCLUDE = os.environ.get("ARMADILLO_INCLUDE", "/opt/homebrew/include")
ARMA_LIB     = os.environ.get("ARMADILLO_LIB", "/opt/homebrew/lib")

setup(
    name="cooltorch",
    version="0.1.0",
    description="PyTorch port of Causes of Outcome Learning (CoOL)",
    packages=find_packages(),
    include_package_data=True,
    package_data={"CoolTorch": ["utils/*.r", "data/*.csv",
                                "tests/R_data/*", "tests/R_pre_trained_model/*"]},
    install_requires=["torch", "numpy", "pandas", "scipy",
                      "scikit-learn", "matplotlib"],
    ext_modules=[
        CppExtension(
            name="cool_ext_arma",
            sources=["CoolTorch/train/cool_step_arma.cpp"],
            include_dirs=[ARMA_INCLUDE],
            library_dirs=[ARMA_LIB],
            libraries=["armadillo"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
