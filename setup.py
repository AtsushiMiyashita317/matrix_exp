from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='matrix_exp',
      ext_modules=[cpp_extension.CppExtension('matrix_exp', ['matrix_exp.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})