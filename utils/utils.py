from ast import literal_eval


def eval_with_exception(string, in_case):
    try:
        return literal_eval(string)
    except ValueError:
        return None
