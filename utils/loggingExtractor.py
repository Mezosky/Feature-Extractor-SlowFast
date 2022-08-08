import logging

class LoggingExtractor():
    def __init__(self,
                filename='extractor_executions.log', 
                encoding='utf-8', 
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', 
                datefmt='%m/%d/%Y %I:%M:%S %p'):
        
        logging.basicConfig(filename=filename, 
                            encoding=encoding, 
                            level=level,
                            format=format, 
                            datefmt=datefmt
                            )

        logging.info('Initializing the Logger')

    def 
