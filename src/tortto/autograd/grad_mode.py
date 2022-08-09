import sys
import functools
import inspect
from typing import Any, Callable, TypeVar, cast


grad_enabled = True

def is_grad_enabled():
    return grad_enabled

FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator"""

    def __call__(self, func: F) -> F:
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)
        return cast(F, decorate_context)

    def _wrap_generator(self, func):
        """Wrap each generator invocation with the context manager"""
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)

            # Generators are suspended and unsuspended at `yield`, hence we
            # make sure the grad mode is properly set every time the execution
            # flow returns into the wrapped generator and restored when it
            # returns through our `yield` to our caller (see PR #49017).
            cls = type(self)
            try:
                # Issuing `None` to a generator fires it up
                with cls():
                    response = gen.send(None)

                while True:
                    try:
                        # Forward the response to our caller and get its next request
                        request = yield response

                    except GeneratorExit:
                        # Inform the still active generator about its imminent closure
                        with cls():
                            gen.close()
                        raise

                    except BaseException:
                        # Propagate the exception thrown at us by the caller
                        with cls():
                            response = gen.throw(*sys.exc_info())

                    else:
                        # Pass the last request to the generator and get its response
                        with cls():
                            response = gen.send(request)

            # We let the exceptions raised above by the generator's `.throw` or
            # `.send` methods bubble up to our caller, except for StopIteration
            except StopIteration as e:
                # The generator informed us that it is done: take whatever its
                # returned value (if any) was and indicate that we're done too
                # by returning it (see docs for python's return-statement).
                return e.value

        return generator_context

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

class set_grad_enabled:
    def __init__(self, mode):
        global grad_enabled
        self.prev = is_grad_enabled()
        grad_enabled = mode

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        global grad_enabled
        grad_enabled = self.prev

class no_grad(_DecoratorContextManager):
    def __init__(self):
        super().__init__()
        self.prev = False

    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)

    def __exit__(self, exc_type, exc_value, traceback):
        set_grad_enabled(self.prev)
