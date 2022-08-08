import logging

log = logging.getLogger(__name__)

class TextYes():
    def __init__(self, a):
        self.a = a
        self._printText()

    def _printText(self):
        a = f"Hubo una vez un gran rey {self.a}"
        log.info("Text output: %s", self.a)