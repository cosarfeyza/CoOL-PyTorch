import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Armadillo (C++ linear algebra) is needed only for the optional C++ training
# backend (cool_ext_arma). Override these if it lives elsewhere, e.g. Intel
# Homebrew uses /usr/local:  ARMADILLO_INCLUDE=/usr/local/include pip install -e .
ARMA_INCLUDE = os.environ.get("ARMADILLO_INCLUDE", "/opt/homebrew/include")
ARMA_LIB     = os.environ.get("ARMADILLO_LIB", "/opt/homebrew/lib")

# The C++ extension is optional: without it, train_CoOL() raises a clear message
# at call time pointing to pytorch_train_CoOL() instead (see train/trainer.py).
# So a missing/broken Armadillo must not break `pip install` for everyone else —
# skip the extension instead of failing the whole install when it can't be built.
_arma_header_found = os.path.isfile(os.path.join(ARMA_INCLUDE, "armadillo"))

ext_modules = []
cmdclass = {}

if _arma_header_found:
    class BuildExtOptional(BuildExtension):
        """Falls back to a pure-Python install if the C++ extension fails to build,
        instead of failing `pip install` for users without a C/C++ toolchain set up."""
        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as e:
                print(f"WARNING: could not build the optional cool_ext_arma C++ "
                      f"extension ({e}). Continuing without it — train_CoOL() will "
                      f"raise a clear error; use pytorch_train_CoOL() instead.")

    ext_modules = [
        CppExtension(
            name="cool_ext_arma",
            sources=["CoolTorch/train/cool_step_arma.cpp"],
            include_dirs=[ARMA_INCLUDE],
            library_dirs=[ARMA_LIB],
            libraries=["armadillo"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        )
    ]
    cmdclass = {"build_ext": BuildExtOptional}
else:
    print(f"NOTE: Armadillo header not found at {ARMA_INCLUDE}/armadillo — skipping "
          f"the optional cool_ext_arma C++ extension. The pure-PyTorch training path "
          f"(pytorch_train_CoOL) still works; train_CoOL() will raise a clear error "
          f"if called. Install Armadillo and re-run `pip install -e .` to enable it, "
          f"or point ARMADILLO_INCLUDE/ARMADILLO_LIB at an existing install.")

setup(
    name="cooltorch",
    version="0.1.0",
    description="PyTorch port of Causes of Outcome Learning (CoOL)",
    packages=find_packages(),
    include_package_data=True,
    package_data={"CoolTorch": ["utils/*.r", "data/*.csv"]},
    install_requires=["torch", "numpy", "pandas", "scipy",
                      "scikit-learn", "matplotlib"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
