import sys
from types import ModuleType
from logger import logging


import logging


def error_message_detail(error: Exception, error_detail: ModuleType) -> str:
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_number}] error message [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error: Exception, error_detail: ModuleType):
        self.error_message = error_message_detail(error, error_detail)
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e: 
        logging.info("Divide by zero", exc_info=True) 
        raise CustomException(e, sys)
