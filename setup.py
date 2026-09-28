import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Armadillo (C++ linear algebra) is needed only for the optional C++ training
# backend (cool_ext_arma). Default search path depends on platform and package
# manager; override with ARMADILLO_INCLUDE / ARMADILLO_LIB if it lives elsewhere,
# e.g. Intel Homebrew uses /usr/local, vcpkg on Windows uses its own triplet dir:
#   ARMADILLO_INCLUDE=C:\vcpkg\installed\x64-windows\include ^
#   ARMADILLO_LIB=C:\vcpkg\installed\x64-windows\lib pip install -e .
if sys.platform == "darwin":
    _DEFAULT_INCLUDE, _DEFAULT_LIB = "/opt/homebrew/include", "/opt/homebrew/lib"
elif sys.platform.startswith("linux"):
    _DEFAULT_INCLUDE, _DEFAULT_LIB = "/usr/include", "/usr/lib"
else:
    # No safe cross-distribution default on Windows (vcpkg/conda paths vary) --
    # rely entirely on the environment variables; the header-existence check
    # below then correctly skips the extension if they are not set.
    _DEFAULT_INCLUDE, _DEFAULT_LIB = "", ""

ARMA_INCLUDE = os.environ.get("ARMADILLO_INCLUDE", _DEFAULT_INCLUDE)
ARMA_LIB     = os.environ.get("ARMADILLO_LIB", _DEFAULT_LIB)

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
        instead of failing `pip install` for users without a working C++ toolchain.
        On failure, self.extensions is cleared (there is only ever this one
        extension in the package) so setuptools never expects an output file that
        was never produced -- letting super().build_extensions() run normally
        first preserves Torch's own preprocessing of extra_compile_args."""
        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as e:
                print(f"WARNING: could not build the optional cool_ext_arma C++ "
                      f"extension ({e}). Continuing without it — train_CoOL() will "
                      f"raise a clear error; use pytorch_train_CoOL() instead.")
                self.extensions = []

    # MSVC (the default Windows compiler for a CppExtension) does not understand
    # GCC/Clang-style flags like -O3/-std=c++20 -- it needs /O2/std:c++20 instead.
    if sys.platform == "win32":
        _cxx_flags = ["/O2", "/std:c++20"]
    else:
        _cxx_flags = ["-O3", "-std=c++20"]

    ext_modules = [
        CppExtension(
            name="cool_ext_arma",
            sources=["CoolTorch/train/cool_step_arma.cpp"],
            include_dirs=[ARMA_INCLUDE],
            library_dirs=[ARMA_LIB],
            libraries=["armadillo"],
            extra_compile_args={"cxx": _cxx_flags},
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
