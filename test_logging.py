import logging
from test_folder.sum import EasyOps
from test_folder.text import  TextYes

logging.basicConfig(filename='example.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p'
                    )


logging.info('Hello!... initializing your logger')
EasyOps(10, 2)
TextYes("Miaus")

#logging.warning('And this, too')
#logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
