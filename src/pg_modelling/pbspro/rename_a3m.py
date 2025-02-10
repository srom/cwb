"""
colafold_search outputs one alignment file per entry in the input fasta. 
Files are named 1.a3m, 2.a3m, ..., i.a3m, ..., N.a3m. "i" is the index of the sequence in the fasta.
This script moves those file to a different location and rename them based on the id of the sequence "i".
"""
import argparse
import logging
from pathlib import Path
import re

from Bio import SeqIO

logger = logging.getLogger()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--a3m_folder', 
        help='Path to folder containing a3m alignments', 
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--fasta', 
        help='Path to reference fasta file', 
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--output_folder', 
        help='Path to output folder', 
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    a3m_folder = args.a3m_folder
    fasta_path = args.fasta
    output_folder = args.output_folder

    logger.info(f'Moving and renaming .a3m files from {a3m_folder} to {output_folder}')

    index_to_sequence_id = {}
    for i, record in enumerate(SeqIO.parse(fasta_path, 'fasta')):
        index_to_sequence_id[i] = record.id

    n = 0
    for f in a3m_folder.glob('*.a3m'):
        if re.match(r'^[0-9]+\.a3m$', f.name) is not None:
            index = int(f.name.replace('.a3m', ''))
            if index in index_to_sequence_id:
                sequence_id = index_to_sequence_id[index]
                f.replace(output_folder / f'{sequence_id}.a3m')
                n += 1

    logger.info(f'Number of files moved: {n:,}')


if __name__ == '__main__':
    main()
