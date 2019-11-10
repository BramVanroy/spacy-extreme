import datetime
import logging
from multiprocessing import Manager, Pool, Process
from os import cpu_count

import spacy
from tqdm import tqdm

logging.basicConfig(datefmt='%d-%b %H:%M:%S',
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler('progress.log'),
                        logging.StreamHandler()
                    ])


class Extreme:
    def __init__(self,
                 chunker,
                 disabled_pipes=None,
                 is_segmented=False,
                 is_tokenized=False,
                 max_tasks_per_child=None,
                 n_workers=None,
                 spacy_model='en_core_web_sm',
                 writer=None):
        self.chunker = chunker

        disabled_pipes = [] if disabled_pipes is None else disabled_pipes
        self.nlp = spacy.load(spacy_model, disable=disabled_pipes)

        if is_segmented:
            self.nlp.add_pipe(self._prevent_sbd, name='prevent-sbd', first=True)

        self.is_tokenized = is_tokenized
        if is_tokenized:
            self.nlp.tokenizer = self.nlp.tokenizer.tokens_from_list

        self.max_tasks_per_child = max_tasks_per_child
        self.writer = writer

        # reserve processes for read (and optionally write)
        n = 2 if writer else 1
        self.n_workers = max(1, cpu_count()-n) if n_workers is None else max(1, n_workers-n)

        self.results_q = None
        self.work_q = None

    def process(self, func):
        start_time = datetime.datetime.now()
        with Manager() as manager:
            self.work_q = manager.Queue(maxsize=self.n_workers * 2)

            # Create a reader and a writer process
            reader_proc = Process(target=self.reader)
            # The reader starts filling up the work_q
            reader_proc.start()

            if self.writer:
                self.results_q = manager.Queue(maxsize=max(self.n_workers * 100, 256))
                writer_proc = Process(target=self.writer.write, args=(self.results_q,))
                writer_proc.start()

            with Pool(self.n_workers, maxtasksperchild=self.max_tasks_per_child) as pool:
                logging.info('Chunking...')
                worker_jobs = []
                while True:
                    # Get work from the working queue
                    work = self.work_q.get()
                    if work == 'done':
                        break

                    chunk_start, chunk_size = work
                    # Apply work to workers
                    worker_jobs.append(pool.apply_async(self.process_batch, (func, chunk_start, chunk_size)))

                logging.info('Done chunking...')

                # After the queue is 'done', the reader can close
                reader_proc.join()
                reader_proc.terminate()

                # get results
                total_n_sentences = 0
                total_n_tokens = 0
                total_chunk_size = 0
                pbar = tqdm(total=100)
                for job in worker_jobs:
                    results, n_sents, n_toks, chunk_size = job.get()

                    total_n_sentences += n_sents
                    total_n_tokens += n_toks
                    total_chunk_size += chunk_size

                    pbar.n = total_chunk_size * 100 // self.chunker.file_end
                    pbar.refresh()

                    yield results
                pbar.close()

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

    def process_batch(self, func, chunk_start, chunk_size):
        batch = self.chunker.get_batch(chunk_start, chunk_size)

        if self.is_tokenized:
            batch = (s.split() for s in batch)

        docs = self.nlp.pipe(batch)

        results = []
        n_sents = 0
        n_toks = 0
        for doc in docs:
            results.append(func(doc))
            n_sents += len(list(doc.sents))
            n_toks += len(doc)

        if self.results_q:
            self.results_q.put(results)

        return results, n_sents, n_toks, chunk_size

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

    def reader(self):
        for chunk_tuple in self.chunker.chunkify():
            self.work_q.put(chunk_tuple)

        self.work_q.put('done')
