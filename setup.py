import os
import torch
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension


def get_extensions():
    ext_dir = './dolphin/utils/extensions'
    extensions = []
    from torch.utils.cpp_extension import (CUDAExtension, CppExtension)
    define_macros = []
    extra_compile_args = {'cxx': []}
    name = 'dolphin.ext'
    os.environ.setdefault('MAX_JOBS', '4')
    main_source = glob.glob(os.path.join(ext_dir, 'src', '*.cpp'))
    cpp_source = glob.glob(os.path.join(ext_dir, 'src', 'cpu', '*.cpp'))
    cuda_source = glob.glob(os.path.join(ext_dir, 'src', 'cuda', '*.cu'))
    sources = main_source + cpp_source

    if torch.cuda.is_available():
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args['nvcc'] = []
        sources += cuda_source
        extension = CUDAExtension
    else:
        extension = CppExtension
    include_path = os.path.abspath(os.path.join(ext_dir, 'include'))
    ext_ops = extension(
        name=name,
        sources=sources,
        include_dirs=[include_path],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)
    return extensions

setup(
    name='dolphin',
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension})