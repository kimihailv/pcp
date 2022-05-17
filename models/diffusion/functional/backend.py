import os

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(name='ball_query',
                extra_cflags=['-O3', '-std=c++17'],
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'ball_query.cpp',
                    'ball_query.cu',
                    'bindings.cpp'
                ]]
                )

__all__ = ['_backend']
