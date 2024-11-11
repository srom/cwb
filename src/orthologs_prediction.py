"""
-----------------------------------------
DEPRECATED - use eggNOG-mapper instead.
-----------------------------------------

Orthologs finder.
Depends on the output of `run_mmseqs_rbh.sh`, which computes reciprocal best hits for all genomes. 
This scripts parses the output of mmseqs RBH to come up with orthologs as follows:
- Load hit pairs into networkX
- Find connected components: nodes in this component are assumed to form an ortholog group
- Assign ID to each component (ortholog ID)
- Output to CSV

This is very basic. Ideally we should run a dedicated software such as eggNOG instead.
"""
import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
import networkx as nx


logger = logging.getLogger()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_rbh', 
        help='Path to mmseqs easy-rbh output (compressed files are handled)',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-o', '--output_folder', 
        help='Path to output folder',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--prefix', 
        help='Prefix for all output files, e.g. Streptococcus_',
        type=str,
        required=False,
        default='',
    )
    args = parser.parse_args()

    input_rbh = args.input_rbh
    output_folder = args.output_folder
    prefix = args.prefix

    if not input_rbh.is_file():
        logger.error(f'Input reciprocal best hits (RBH) file does not exist: {args.input_rbh}')
        sys.exit(1)
    elif not output_folder.is_dir():
        logger.error(f'Output folder does not exist: {args.output_folder}')
        sys.exit(1)

    logger.info(f'Finding ortholog groups from MMseqs Reciprocal Best Hits output: {input_rbh.as_posix()}')

    logger.info('Parse reciprocal best hits into graph')
    G, node_id_map = parse_hits_into_graph(input_rbh)

    logger.info('Gather ortholog groups')
    ortholog_groups = []
    ortholog_metadata_dict = {
        'ortholog_candidate_id': [],
        'n_genomes': [],
        'n_proteins': [],
    }
    n = 0
    for c in nx.connected_components(G):
        protein_ids = [node_id_map[p] for p in sorted(frozenset(c))]

        n_genomes = len({p.split('@')[1] for p in protein_ids})
        n_proteins = len(protein_ids)

        if n_genomes >= 2 and n_proteins >= 2:
            n += 1
            ortholog_candidate_id = f'ortholog_candidate_{str(n).zfill(5)}'
            ortholog_groups.append(
                (ortholog_candidate_id, protein_ids)
            )
            ortholog_metadata_dict['ortholog_candidate_id'].append(ortholog_candidate_id)
            ortholog_metadata_dict['n_genomes'].append(n_genomes)
            ortholog_metadata_dict['n_proteins'].append(n_proteins)

    # Free memory
    G = None

    ortholog_metadata = pd.DataFrame.from_dict(ortholog_metadata_dict).sort_values(
        ['n_genomes', 'n_proteins', 'ortholog_candidate_id'],
        ascending=[False, False, True],
    )
    ortholog_metadata['ortholog_id'] = [f'ortholog_{str(i+1).zfill(5)}' for i in range(len(ortholog_metadata))]
    id_map = {
        tpl.ortholog_candidate_id: tpl.ortholog_id
        for tpl in ortholog_metadata.itertuples()
    }

    logger.info(f'Number of ortholog groups found: {len(ortholog_groups):,}')

    logger.info('Flatten groups into DataFrame')
    ortholog_df = pd.DataFrame(
        [
            (ortholog_candidate_id, protein_id) 
            for ortholog_candidate_id, protein_ids in ortholog_groups
            for protein_id in protein_ids
        ],
        columns=['ortholog_candidate_id', 'id']
    )
    ortholog_df['ortholog_id'] = ortholog_df['ortholog_candidate_id'].apply(lambda id_: id_map[id_])

    logger.info(f'Export to folder: {output_folder.as_posix()}')
    ortholog_df[
        ['ortholog_id', 'id']
    ].sort_values(
        ['ortholog_id', 'id']
    ).to_csv(output_folder / f'{prefix}ortholog_protein_map.csv', index=False)

    ortholog_metadata[
        ['ortholog_id', 'n_genomes', 'n_proteins']
    ].to_csv(output_folder / f'{prefix}ortholog_metadata.csv', index=False)

    logger.info('DONE')
    sys.exit(0)


def parse_hits_into_graph(input_rbh):
    G = nx.Graph()

    chunk_size = 1_000_000
    chunk_reader = pd.read_csv(input_rbh, sep='\t', header=None, usecols=[0, 1], chunksize=chunk_size)

    node_id_map = {}
    reversed_node_id_map = {}
    n = 0
    for i, chunk in enumerate(chunk_reader):
        logger.info(f'Chunk {i+1}...')

        chunk.columns = ['id_A', 'id_B']
        edges = []
        for a, b in zip(chunk['id_A'], chunk['id_B']):
            if a == b:
                continue

            if a in reversed_node_id_map:
                id_a = reversed_node_id_map[a]
            else:
                id_a = n
                node_id_map[n] = a
                reversed_node_id_map[a] = n
                n += 1
            
            if b in reversed_node_id_map:
                id_b = reversed_node_id_map[b]
            else:
                id_b = n
                node_id_map[n] = b
                reversed_node_id_map[b] = n
                n += 1

            edges.append((id_a, id_b))

        if len(edges) > 0:
            G.add_edges_from(edges)

    return G, node_id_map


if __name__ == '__main__':
    main()
