import numpy as np
import pytest


def raises_exception(exception, fun):
    if exception is not None and issubclass(exception, Exception):
        with pytest.raises(exception):
            fun()
    else:
        return fun()


# use pytest.raises(... match="message") insted
def assert_eq_err_message(record, message):
    assert 0
    assert record.value.args[0] == message


# pytest.wars(... match="message") is funky in some cases
def assert_eq_warn_message(record, message):
    assert record[0].message.args[0] == message


def squarediff(a, b):
    return np.sum((np.asarray(a) - np.asarray(b)) ** 2)
