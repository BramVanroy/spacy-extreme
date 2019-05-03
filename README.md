# spacy-extreme
An example of how to use spaCy for extremely large files without running into memory issues

## Memory issues with spaCy
SpaCy is a popular, powerful NLP tool that can process a text and get almost any information out of it that you could need. 
Unfortunately I started running into issues when multiprocessing a single file of 30GB+: the memory usage kept growing. 
Even with [the simplest base case](https://github.com/explosion/spaCy/issues/3618) the issue persists. 
A 'bug fix' is not available, because it is not clear where the memory is leaking. One would expect that the issue lies 
in spaCy itself, but that would imply that reloading a spaCy instance should free that memory.
But that is [not the case](https://github.com/explosion/spaCy/issues/3618#issuecomment-485832596).
It is hard, then, to find a fix - because it is unclear where to start looking.

Because of that, I figured that there must be another way.
The solution lies in the `multiprocessing` library, and more specifically in one of the parameters for 
[`Pool`](https://docs.python.org/3.7/library/multiprocessing.html#multiprocessing.pool.Pool).
`maxtasksperchild` is a parameter that ensures that a single child process will execute only n tasks. After that, it will
be killed, its memory freed, and replaced by a new process.
That is exactly what we need! 
The memory grows because more and more data is read by a process. We want to limit the number of batches that a process
can process so that its memory usage is being kept in check.

## Parsing huge files: how to be lenient on memory?
Another issue that you may be faced with, is processing an enormous file and distributing it over child processes,
without running into memory issues.
We want to process these large files in batches, which will make processing more efficient.
These batches cannot be too small because then the workers will consume the batches too quickly,
causing only a few workers to be actively processing batches at a time.
In the example code, you will find a
[`Chunker`](https://github.com/BramVanroy/spacy-extreme/blob/master/main.py#L47-L68) class.
This chunker will retrieve *file pointers* from a file. These are integers representing a position in a file, you can
think of it as the cursor position, in bytes.
In every step, the cursor moves forward `batch_size` bytes, and return the position of the cursor.
When the child process retrieves a cursor position, it will look it up in the file, and get a `batch_size`d chunk.
This chunk can then be processed.
As may be clear, the actual file contents are *not* retrieved by the first step in the reader process.
We do not want to share these huge chunks of data between processes, but the file pointer is just an integer; easily and quickly shared.

Parsing a file in chunks has some shortcomings, as a chunk does not necessarily end at a line ending,
which leaves you with broken sentences.
To remedy that, the first and last lines of each batch are retrieved and kept separately, as they are likely to be broken.
As a last step, these broken lines are stitched back together and parsed as a single batch.

For a large file_size/batch_size ratio, this may cause to large memory consumption as well, but I have not encountered this issue yet.
A solution would be to periodically check for new 'partials', as I call broken sentences, and stitch those together; or add these 
partials to the results queue separately and let the writer process them when available.
