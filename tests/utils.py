def assert_eq_err_message(record, message):
    assert record.value.args[0] == message
