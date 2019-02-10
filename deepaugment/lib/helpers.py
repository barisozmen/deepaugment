def log_and_print(text, logging_obj, level="info"):
    print(text)
    if level == "info":
        logging_obj.info(text)
    elif level == "warning":
        logging_obj.warning(text)
    elif level == "error":
        logging_obj.error(text)
