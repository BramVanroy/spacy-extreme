import datetime
from html import unescape
import logging
from math import inf
from multiprocessing import Manager, Pool, Process
from os import cpu_count, stat
from pathlib import Path
import re

import psutil
import spacy

logging.basicConfig(datefmt='%d-%b %H:%M:%S',
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler('progress.log'),
                        logging.StreamHandler()
                    ])

DEFAULT_WORKERS = (cpu_count() - 1) or 1

""" Processes a single, huge text file with spaCy, without running into memory issues 
    IF the right parameters are chosen.
    
    Important parameters:
        - -b, --batch-size: the batch size (in bytes) to process at the same time. A larger batch, will mean that
            every task (in the max-tasks-per-child) will use more memory. You need to find a good balance between
            batch-size and max-tasks-per-child.
        - -m, --max-tasks-per-child: the number of batches to process before a child process is killed and replaced
            this will effectively free the memory used by that child process. If you are very low on memory, 
            set this to 1, meaning that each process will only process one batch before being replaced.
        - -n, --n-workers: the number of child processes to spawn that will process the batches. It is important
            to know that the readers and writers are working in their own subprocesses, so don't use all cores 
            for n-workers. Also, the more cores you put to work simultaneously, the more memory you will be using.
            On top of that, if your batch-size is too small, the reader will not be fast enough to feed all the workers.
            So, again, you need to find a good trade-off focused on the batch-size.
        - --space-model: it makes sense that if you use a large spaCy model, you will consume more memory. 

    Reading input happens in chunks. The byte file pointers of each chunk are passed to the child processes,
    leaving them in charge of actually getting the contents from the file.
    Because byte chunks are line-agnostic, we assume that the last line of each chunk is an incomplete line whose
    second part is actually the first line of the next chunk. Therefore, we return the first and last line of all
    chunks, and process them at the very end; stitching them back together.
    This means, though, that the order of the sentence in the input file is NOT preserved.
    
    You can use this file as a template, and only change the process_batch method to your liking.
    That's where the actual values from spaCy are retrieved and processed.
"""


class Chunker:
    def __init__(self, pfin, batch_size=1024 ** 2):
        self.batch_size = batch_size
        self.pfin = pfin

        logging.info(f"Chunking with a batch size of {batch_size:,} bytes.")

    def chunkify(self):
        """  Yields a tuple containing the starting file pointer to a new batch,
             and whether a batch is the first batch, or the last batch. """
        file_end = stat(self.pfin).st_size

        # If the file is smaller than or equal to the buffer size,
        # we can get it all in one batch
        if file_end <= self.batch_size:
            yield 0, True, True
            return None
        else:
            for chunk_start in range(0, file_end - self.batch_size, self.batch_size):
                yield chunk_start, chunk_start == 0, False

            yield chunk_start + self.batch_size, False, True

    def get_batch(self, chunk_start, is_first, is_last):
        with open(self.pfin, 'rb') as f:
            f.seek(chunk_start)
            chunk = f.read(self.batch_size)

        batch = chunk.split(b'\n')

        # Only pop the first/last line if the batch is not
        # the first/last batch respectively
        first_line = batch.pop(0) if not is_first else None
        last_line = batch.pop(-1) if not is_last else None

        return [s.decode('utf-8').rstrip() for s in filter(None, batch)], first_line, last_line, chunk_start


class Representator:
    def __init__(self,
                 do_html=False,
                 do_unicode=False,
                 do_lowercase=False,
                 normalize_url=False,
                 normalize_digits=False,
                 max_length=None,
                 min_length=None,
                 spacy_model='en_core_web_sm'):

        self.max_length = max_length if max_length else inf
        self.min_length = min_length if min_length else 0

        self.nlp = spacy.load(spacy_model, disable=['ner', 'textcat'])
        self.nlp.add_pipe(self._prevent_sbd, name='prevent-sbd', before='parser')
        self.tagmap = self.nlp.Defaults.tag_map

        self.results_q = None
        self.work_q = None

        self.chunker = None

        self.do_html = do_html
        self.do_unicode = do_unicode
        self.do_lowercase = do_lowercase
        self.do_normalize_url = normalize_url
        self.do_normalize_digits = normalize_digits

        self.digit_table = str.maketrans("0123456789", "1111111111")
        self.url_regex = r'((([A-Za-z]{3,9}:(?:\/\/)?)(?:[-;:&=\+\$,\w]+@)?[A-Za-z0-9.-]+|(?:www\d*\.|[-;:&=\+\$,\w]+@)[A-Za-z0-9.,)(-]+)((?:\/[\+~%\/.\w_-]*)?\??(?:[-\+=&;%@.\w_]*)#?(?:[\w]*))?)'
        self.unicode_regex = r'(?<!\b[a-zA-Z]:)(\\u[0-9A-Fa-f]{4})'

    def process(self, pfin, n_workers, max_tasks_per_child):
        logging.info(f"Started processing {pfin.name} with {n_workers} workers.")
        if max_tasks_per_child:
            logging.info(f"Max. {max_tasks_per_child} tasks per child process before replacement.")

        start_time = datetime.datetime.now()

        total_n_sentences = 0
        total_n_tokens = 0
        with Manager() as manager:
            self.results_q = manager.Queue(maxsize=max(n_workers * 100, 256))
            self.work_q = manager.Queue(maxsize=n_workers * 2)

            # Create a reader and a writer process
            reader_proc = Process(target=self.reader)
            # The reader starts filling up the work_q
            reader_proc.start()
            writer_proc = Process(target=self.writer, args=(pfin,))
            writer_proc.start()

            with Pool(n_workers, maxtasksperchild=max_tasks_per_child) as pool:
                worker_jobs = []
                partials = []

                while True:
                    # Get work from the working queue
                    work = self.work_q.get()
                    if work == 'done':
                        break

                    batch_start, is_first_batch, is_last_batch = work
                    # Apply work to workers
                    job = pool.apply_async(self.process_batch, (batch_start, is_first_batch, is_last_batch))
                    worker_jobs.append(job)

                # After the queue is 'done', the reader can close
                reader_proc.join()
                reader_proc.terminate()

                # When a worker has finished its job, get its information back
                for job_idx, job in enumerate(worker_jobs, 1):
                    n_sentences, n_tokens, first_line, last_line, chunk_start = job.get()

                    partials.append((chunk_start, first_line, last_line))
                    total_n_sentences += n_sentences
                    total_n_tokens += n_tokens

                    # Log some progress info
                    if job_idx == 1 or job_idx % n_workers == 0:
                        time_since_start = (datetime.datetime.now() - start_time)
                        sents_perf = total_n_sentences // time_since_start.total_seconds()
                        time_since_start = self._format_time(time_since_start)
                        logging.info(f"Processed batch #{job_idx:,}: {n_sentences:,} sents ({sents_perf:,.0f} sents/s)."
                                     f" Mem. use: {psutil.virtual_memory().percent}%. Running for {time_since_start}")

                # Process all partials as a single batch
                # When your batch size is very small, this will give a lot of overhead
                n_sentences, n_tokens, _, _, _ = self.process_last_batch(partials)
                total_n_sentences += n_sentences
                total_n_tokens += n_tokens

                # Notify the writer that we're done
                self.results_q.put('done')

            writer_proc.join()
            writer_proc.terminate()

        # Log some info
        running_time = (datetime.datetime.now() - start_time)
        sents_perf = total_n_sentences // running_time.total_seconds()
        running_time = self._format_time(running_time)
        logging.info(f"Done processing in {running_time} ({sents_perf:,.0f} sentences/s)."
                     f" Processed {total_n_sentences:,.0f} sentences and {total_n_tokens:,.0f} tokens.")

    def process_last_batch(self, partials):
        # Sort partials on their start-byte to ensure
        # that they are ordered correctly
        partials = sorted(partials, key=lambda x: x[0])

        batch = []
        prev_last = None
        for _, first, last in partials:
            if prev_last is not None and first is not None:
                chunk = prev_last + first
                batch.append(chunk.decode('utf-8'))

            prev_last = last

        return self.process_batch(batch=batch)

    def process_batch(self, chunk_start=None, is_first=None, is_last=None, batch=None):
        # 'batch' is always None, except for the very last batch which processes all partials.
        # See 'process_last_batch'.
        if batch is None:
            batch, first_line, last_line, chunk_start = self.chunker.get_batch(chunk_start, is_first, is_last)
        else:
            first_line = None
            last_line = None
            chunk_start = None

        # Might make more sense to do this immediatelly in the chunker
        # But that would make processing annoying for the last batch
        if self.do_html:
            batch = map(unescape, batch)
        if self.do_unicode:
            batch = map(self.unicode_replace, batch)
        if self.do_normalize_url:
            batch = map(self.normalize_url, batch)
        if self.do_normalize_digits:
            batch = map(self.normalize_digits, batch)

        # Parse text with spaCy
        docs = list(self.nlp.pipe(batch))
        # Chop into sentences
        spacy_sents = [sent for doc in docs for sent in doc.sents]
        # Filter too long or too short sentences
        spacy_sents = [sent for sent in spacy_sents if self.min_length <= len(sent) <= self.max_length]
        n_sentences = len(spacy_sents)
        n_tokens = 0

        # Get some value from spaCy that we want to write to files
        # Here we just get the tokens, but you can change it to whatever you want
        sents_tok = []
        for sent in spacy_sents:
            n_tokens += len(sent)
            if self.do_lowercase:
                sents_tok.append(' '.join([token.text.lower() for token in sent]))
            else:
                sents_tok.append(' '.join([token.text for token in sent]))

        # Pass results to queue, so they can be written to file by the writer
        self.results_q.put(sents_tok)

        # Return the number of sentences and number of tokens, just to keep track
        # Also return first and last line. These are likely to be 'broken' sentences
        # due to chunking. After processing everything, we will process these 'partial
        # sentences' separately in the main process.
        return n_sentences, n_tokens, first_line, last_line, chunk_start

    def normalize_url(self, line, repl='@url@'):
        return re.sub(self.url_regex, repl, line)

    def normalize_digits(self, line):
        return line.translate(self.digit_table)

    def unicode_replace(self, line):
        def repl(match):
            match = match.group()
            try:
                return match.encode('utf-8').decode('unicode-escape')
            except UnicodeDecodeError:
                return match

        return re.sub(self.unicode_regex, repl, line)

    @staticmethod
    def _prevent_sbd(doc):
        # If you already have one sentence per line in your file
        # you may wish to disable sentence segmentation with this function,
        # which is added to the nlp pipe in the constructor
        for token in doc:
            token.is_sent_start = False
        return doc

    @staticmethod
    def _format_time(delta):
        hours, remainder = divmod(delta.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{hours:02,.0f}:{minutes:02.0f}:{seconds:02.0f}"

    # I/O methods
    def writer(self, pfin):
        with open(pfin.with_suffix('.out'), 'w', encoding='utf-8') as fhout:
            while True:
                m = self.results_q.get()
                if m == 'done':
                    break

                fhout.write('\n'.join(m) + '\n')
                fhout.flush()

    def reader(self):
        for chunk_tuple in self.chunker.chunkify():
            self.work_q.put(chunk_tuple)

        self.work_q.put('done')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parse HUGE text files with spaCy in parallel without running'
                                                 ' into memory issues.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fin', help='input file.')

    parser.add_argument('--do-html', action='store_true', help='unescape HTML characters.')
    parser.add_argument('--do-unicode', action='store_true', help='convert unicode.')
    parser.add_argument('--do-lowercase', action='store_true', help='lower case the text.')
    parser.add_argument('--normalize-digits', action='store_true', help="replace all digits by '1'.")
    parser.add_argument('--normalize-url', action='store_true', help="replace URLs by a '@url@' token.")

    parser.add_argument('-b', '--batch-size', type=int, default=1024 ** 2,
                        help='batch size (in bytes).')
    parser.add_argument('--max-length', type=int, default=None,
                        help="sentences with more than 'max_length' will not be included in the output.")
    parser.add_argument('-m', '--max-tasks-per-child', type=int, default=5,
                        help="max number of batches that a child process can process before it is killed and replaced."
                             " Use this when running into memory issues.")
    parser.add_argument('--min-length', type=int, default=None,
                        help="sentences with less than 'min_length' will not be included in the output.")
    parser.add_argument('-n', '--n-workers', type=int, default=DEFAULT_WORKERS,
                        help=f"number of workers to use (default depends on your current system).")
    parser.add_argument('--spacy-model', default='en_core_web_sm',
                        help='spaCy model to use (must be installed).')
    args = parser.parse_args()

    args = vars(args)
    file_in = Path(args.pop('fin')).resolve()
    workers = args.pop('n_workers')
    b_size = args.pop('batch_size')
    max_tasks = args.pop('max_tasks_per_child')

    representer = Representator(**args)
    representer.chunker = Chunker(file_in, b_size)
    representer.process(file_in, workers, max_tasks)
