from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='linear_cpp',
      ext_modules=[cpp_extension.CppExtension(name = 'linear_cpp', 
                sources = ['../src/linear.cpp', 
                			'../src/block_sizing.cpp', 
                            '../src/cake_dgemm.cpp', 
                            '../src/pack.cpp', 
                            '../src/util.cpp', 
                            '../src/unpack.cpp'],
                extra_compile_args = ['-O3','-Wall','-Wno-unused-function','-Wfatal-errors', 
                '-fopenmp', '-fPIC','-D_POSIX_C_SOURCE=200112L', '-DBLIS_VERSION_STRING=\"0.8.0-13\"'],
                extra_link_args = ['/usr/local/lib/libblis.a', '-lm', '-lpthread', '-lrt'],
				include_dirs = ['.', '/usr/local/include/blis'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

