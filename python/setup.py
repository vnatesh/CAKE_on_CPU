from setuptools import setup, Extension
from torch.utils import cpp_extension
from pathlib import Path
import os

cwd = os.getcwd()
path = Path(cwd)
path = str(path.parent) + '/src'


setup(name='linear_cpp',
      ext_modules=[cpp_extension.CppExtension(name = 'linear_cpp', 
                sources = ['%s/linear.cpp' % path, 
                			'%s/block_sizing.cpp' % path, 
                            '%s/cake_dgemm.cpp' % path, 
                            '%s/pack.cpp' % path, 
                            '%s/util.cpp' % path, 
                            '%s/unpack.cpp' % path],
                extra_compile_args = ['-O3','-Wall','-Wno-unused-function','-Wfatal-errors', 
                '-fopenmp', '-fPIC','-D_POSIX_C_SOURCE=200112L', '-DBLIS_VERSION_STRING=\"0.8.0-13\"'],
                extra_link_args = ['/usr/local/lib/libblis.a', '-lm', '-lpthread', '-lrt'],
				include_dirs = ['.', '/usr/local/include/blis'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

