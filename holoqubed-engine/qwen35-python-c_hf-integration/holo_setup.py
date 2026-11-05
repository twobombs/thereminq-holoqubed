from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='holo_ext',
    ext_modules=[
        CppExtension(
            name='holo_ext',
            sources=['holo_ext.cpp'],
            libraries=['OpenCL'], # Link the OpenCL driver dynamically
            extra_compile_args=['-O3'] # Max C++ optimization
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
