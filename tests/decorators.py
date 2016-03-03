import arrayfire

def foreach_backend(func):
    def wrapper(*args, **kws):
        for backend in arrayfire.library.get_available_backends():
            if backend == 'opencl':
                continue
            arrayfire.library.set_backend(backend)
            func(*args, **kws)
    return wrapper
