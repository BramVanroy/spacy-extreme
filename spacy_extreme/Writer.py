from pathlib import Path


class Writer:
    def __init__(self, fout, func=None):
        self.pfout = Path(fout)
        self.func = func

    def write(self, queue):
        with self.pfout.open('w', encoding='utf-8') as fhout:
            while True:
                data = queue.get()
                if data == 'done':
                    break

                if self.func:
                    data = self.func(data)
                else:
                    data = ''.join(data) + '\n'

                fhout.write(data)
                fhout.flush()
