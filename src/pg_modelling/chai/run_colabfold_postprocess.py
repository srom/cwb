import argparse
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)-10s (%(levelname)s) %(message)s')

    parser = argparse.ArgumentParser(description='Chai postprocessing of ColabFold MSAs')
    parser.add_argument('msa_dir', type=Path, help='Directory where the MSAs from ColabFold are saved')
    args = parser.parse_args()
    msa_dir = args.msa_dir

    logger.info('Converting MSA files to Chai-compatible parquet files')

    if not msa_dir.is_dir():
        logger.error(f'MSA directory does not exist: {msa_dir}')
        sys.exit(1)

    output_dir = msa_dir / 'chai_msa'
    output_dir.mkdir(exist_ok=True)

    for msa_file in msa_dir.glob('*.a3m'):
        logger.info(f'Converting MSA file: {msa_file}')
        with tempfile.TemporaryDirectory() as msa_tempdir:
            a3m_tmp_file = Path(msa_tempdir) / msa_file.name
            shutil.copy(msa_file, a3m_tmp_file)

            res = subprocess.run(
                [
                    'chai', 'a3m-to-pqt', msa_tempdir,
                    '--output-directory', output_dir.resolve().as_posix(),
                ],
                stdout=sys.stdout, 
                stderr=sys.stderr,
            )
            returncode = res.returncode

        if returncode != 0:
            sys.exit(returncode)

    logger.info('MSA files converted to Chai-compatible parquet files')
    sys.exit(0)


if __name__ == '__main__':
    main()
