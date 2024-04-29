import sys

def error_message_detail(error_message: str, error_detail: sys) -> str:
    """Create a custom error message"""
    filename = error_detail.exc_info()[2].tb_frame.f_code.co_filename
    lineno = error_detail.exc_info()[2].tb_lineno

    custom_message = "Error occured in file [{0}] at line number [{1}]. Error message: [{2}]".format(filename, lineno, error_message)

    return custom_message


class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
    