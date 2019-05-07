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

## Usage

```bash
usage: main.py [-h] [-b BATCH_SIZE] [--max-length MAX_LENGTH]
               [-m MAX_TASKS_PER_CHILD] [--min-length MIN_LENGTH]
               [-n N_WORKERS] [--spacy-model SPACY_MODEL]
               fin

Parse HUGE text files with spaCy in parallel without running into memory
issues.

positional arguments:
  fin                   input file.

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size (in bytes). (default: 1048576)
  --max-length MAX_LENGTH
                        sentences with more than 'max_length' will not be
                        included in the output. (default: None)
  -m MAX_TASKS_PER_CHILD, --max-tasks-per-child MAX_TASKS_PER_CHILD
                        max number of batches that a child process can process
                        before it is killed and replaced. Use this when
                        running into memory issues. (default: 5)
  --min-length MIN_LENGTH
                        sentences with less than 'min_length' will not be
                        included in the output. (default: None)
  -n N_WORKERS, --n-workers N_WORKERS
                        number of workers to use (default depends on your
                        current system).
  --spacy-model SPACY_MODEL
                        spaCy model to use (must be installed). (default:
                        en_core_web_sm)
```

## Best settings
It is hard to tell what the best settings are for a given combination of hardware and the data.
On a machine with 384GB of memory and 48 cores, I ran the script with the following settings.
Memory consumption never exceeded 78%.

- `-n 24`: using 24 cores. 
- `--spacy-model en_core_web_lg`: the largest Englsih spaCy model
- `-b 50000000`: a batch size of 50 MB (50,000,000 bytes). With my data, this was roughly equivalent to 400k sentences
- `-m 5`: replace a process after having processed 5 batches. In total each process processes 2M sentences before being replaced

If you do not have a lot of memory available, you will want to set `--max-tasks-per-child` (`-m`) to 1 so that an active process is replaced after each batch.
In such case, ensure that your batch size is not too small (e.g. not less than 100kB) to maximize efficiency. 
