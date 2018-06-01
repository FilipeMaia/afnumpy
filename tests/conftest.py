import arrayfire
import pytest

backends = arrayfire.library.get_available_backends()
# do not use opencl backend, it's kinda broken
#backends = [x for x in backends if x != 'opencl']

# This will set the different backends before each test is executed
@pytest.fixture(scope="function", params=backends, autouse=True)
def set_backend(request):
    arrayfire.library.set_backend(request.param, unsafe=True)
