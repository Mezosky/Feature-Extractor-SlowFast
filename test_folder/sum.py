import logging

log = logging.getLogger(__name__)

class EasyOps():
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self._plusUltra()

    def _plusUltra(self):
        suma = self.a + self.b
        log.info('sum: %s', suma)
    
    def _plusPot(self):
        pot = self.a**self.b
        log.info('pot: %s', pot)


