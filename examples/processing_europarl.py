from pathlib import Path

from spacy_extreme import Chunker, Extreme, Writer


def posify(doc):
    posified = '\n'.join([' '.join([tok.pos_ for tok in sent]) for sent in doc.sents]) + '\n'

    return posified


if __name__ == '__main__':
    fin = r'C:/dev/data/europarl-v9.en'
    chunker = Chunker(fin, batch_size=10000)
    writer = Writer(Path(fin).with_suffix('.pos.en'))
    xtreme = Extreme(chunker, is_segmented=True, disabled_pipes=['ner', 'parser'], writer=writer)

    _ = list(xtreme.process(posify))
