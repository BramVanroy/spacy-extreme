import logging
from os import stat
from pathlib import Path

from typing import Generator, Union

logger = logging.getLogger(__name__)


class Chunker:
    """ Chunker that can chunk a file into byte ranges which can then be retrieved as a list of encoded lines. """
    def __init__(self, fin: Union[str, bytes, Path], batch_size: int = 1000, encoding: str = 'utf-8'):
        """
        :param fin: filename to chunk
        :param batch_size: approximate size of each chunk (in kilobytes)
        :param encoding: encoding of the input file. Will be used when retrieving the encoded batch of lines
        """
        self.batch_size = int(batch_size * 1e3)
        self.encoding = encoding
        self.pfin = Path(fin).resolve()

        logger.info(f"Chunking with a batch size of {batch_size:,} kilobytes.")

    def chunkify(self) -> Generator:
        """ Chunks a file into sequential byte ranges of approximately the same size as defined in the constructor.
        The size of each chunk is not exactly the same because if a chunk ends on an incomplete line, the remainder
        of the line will also be read and included in the chunk.

        :returns a generator that yields tuples of two integers: the starting byte of the chunk and its size
        """
        file_end = stat(self.pfin).st_size

        # If the file is smaller than or equal to the buffer size, we can get it all in one batch
        if file_end <= self.batch_size:
            yield 0, file_end
        else:
            with self.pfin.open('rb') as fhin:
                prev_pos = 0
                while prev_pos < file_end:
                    pos = prev_pos + self.batch_size
                    fhin.seek(pos)
                    fhin.readline()
                    pos = fhin.tell()
                    yield prev_pos, pos - prev_pos
                    prev_pos = pos

    def get_batch(self, chunk_start: int, chunk_size: int, rm_newlines: bool = True) -> Generator:
        """ Retrieves a chunk, given a starting byte and chunk size, as a batch of encoded lines through a generator.

        :param chunk_start: the starting byte position of the requested chunk
        :param chunk_size: the size of the requested chunk
        :param rm_newlines: whether to remove the newlines at the end of each line (rstrip)
        :returns a generator that yields each encoded line in the batch
        """
        with open(self.pfin, 'rb') as f:
            f.seek(chunk_start)
            chunk = f.read(chunk_size)

        if rm_newlines:
            return (s.decode(self.encoding).rstrip() for s in chunk.split(b'\n') if s)
        else:
            return (s.decode(self.encoding) for s in chunk.split(b'\n') if s)
