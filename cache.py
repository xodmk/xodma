#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function caching"""

import os
import sys
from typing import Any, Callable, TypeVar
from joblib import Memory
from decorator import FunctionMaker


def _decorator_apply(dec, func):
    return FunctionMaker.create(
        func,
        "return decfunc(%(shortsignature)s)",
        dict(decfunc=dec(func)),
        __wrapped__=func,
    )


_F = TypeVar("_F", bound=Callable[..., Any])

class CacheManager(object):
    """The librosa cache manager class wraps joblib.Memory
    with a __call__ attribute, so that it may act as a function.
    Additionally, it provides a caching level filter, so that
    different functions can be cached or not depending on the user's
    preference for speed vs. storage usage.
    """

    def __init__(self, *args: Any, **kwargs: Any):

        level = kwargs.pop("level", 10)

        # Initialize the memory object
        self.memory: Memory = Memory(*args, **kwargs)
        # The level parameter controls which data we cache
        # smaller numbers mean less caching
        self.level: int = level

    def __call__(self, level: int) -> Callable[[_F], _F]:
        """Example usage:
        @cache(level=2)
        def semi_important_function(some_arguments):
            ...
        """

        def wrapper(function):
            """Decorator function.  Adds an input/output cache to
            the specified function."""
            if self.memory.location is not None and self.level >= level:
                return _decorator_apply(self.memory.cache, function)

            else:
                return function

        return wrapper

    def clear(self, *args: Any, **kwargs: Any) -> None:
        return self.memory.clear(*args, **kwargs)

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        return self.memory.eval(*args, **kwargs)

    def format(self, *args: Any, **kwargs: Any) -> str:
        return self.memory.format(*args, **kwargs)

    def reduce_size(self, *args: Any, **kwargs: Any) -> None:
        return self.memory.reduce_size(*args, **kwargs)

    def warn(self, *args: Any, **kwargs: Any) -> None:
        return self.memory.warn(*args, **kwargs)


# Instantiate the cache from the environment
cache: CacheManager = CacheManager(
    os.environ.get("LIBROSA_CACHE_DIR", None),
    mmap_mode=os.environ.get("LIBROSA_CACHE_MMAP", None),
    compress=os.environ.get("LIBROSA_CACHE_COMPRESS", False),
    verbose=int(os.environ.get("LIBROSA_CACHE_VERBOSE", 0)),
    level=int(os.environ.get("LIBROSA_CACHE_LEVEL", 10)),
)

# Override the module's __call__ attribute
sys.modules[__name__] = cache




# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """Function caching"""
#
# import os
# import sys
# from joblib import Memory
#
#
# class CacheManager(Memory):
#     """The librosa cache manager class extends joblib.Memory
#     with a __call__ attribute, so that it may act as a function.
#
#     This allows us to override the librosa.cache module's __call__
#     field, thereby allowing librosa.cache to act as a decorator function.
#     """
#
#     def __init__(self, cachedir, level=10, **kwargs):
#         super(CacheManager, self).__init__(cachedir, **kwargs)
#         # The level parameter controls which data we cache
#         # smaller numbers mean less caching
#         self.level = level
#
#     def __call__(self, level):
#         """Example usage:
#
#         @cache(level=2)
#         def semi_important_function(some_arguments):
#             ...
#         """
#         def wrapper(function):
#             """Decorator function.  Adds an input/output cache to
#             the specified function."""
#
#             from decorator import FunctionMaker
#
#             def decorator_apply(dec, func):
#                 """Decorate a function by preserving the signature even if dec
#                 is not a signature-preserving decorator.
#
#                 This recipe is derived from
#                 http://micheles.googlecode.com/hg/decorator/documentation.html#id14
#                 """
#
#                 return FunctionMaker.create(
#                     func, 'return decorated(%(signature)s)',
#                     dict(decorated=dec(func)), __wrapped__=func)
#
#             if self.cachedir is not None and self.level >= level:
#                 return decorator_apply(self.cache, function)
#
#             else:
#                 return function
#         return wrapper
#
#
# # Instantiate the cache from the environment
# CACHE = CacheManager(os.environ.get('LIBROSA_CACHE_DIR', None),
#                      mmap_mode=os.environ.get('LIBROSA_CACHE_MMAP', None),
#                      compress=os.environ.get('LIBROSA_CACHE_COMPRESS', False),
#                      verbose=int(os.environ.get('LIBROSA_CACHE_VERBOSE', 0)),
#                      level=int(os.environ.get('LIBROSA_CACHE_LEVEL', 10)))
#
# # Override the module's __call__ attribute
# sys.modules[__name__] = CACHE
