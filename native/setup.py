from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#run this file with python setup.py build_ext --inplace
# it will build and install the gile
setup(
    name="lru_native_extensions",
    ext_modules=[
        CUDAExtension(
            name="lru_scan_forward",
            sources=['src/extension.cpp', 'src/kernel.hip'], 

        ),
        CUDAExtension(
            name="lru_scan_backward",
            sources=['src/extensionbackward.cpp', 'src/kernelbackward.hip'], 
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
