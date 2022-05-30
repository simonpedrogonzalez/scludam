import pytest

def raises_exception(exception, fun):
    if exception is not None and issubclass(exception, Exception):
        with pytest.raises(exception):
            fun()
    else:
        return fun()
        
def assert_eq_err_message(record, message):
    assert record.value.args[0] == message
