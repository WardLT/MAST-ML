__all__ = ['check_paths']

from datetime import datetime
import logging
import os

log = logging.getLogger('mastml')


def check_paths(conf_path, data_path, outdir):
    # Check conf path:
    if os.path.splitext(conf_path)[1] != '.conf':
        raise utils.FiletypeError(
                                  f"Conf file does not end"
                                  f"in .conf: '{conf_path}'"
                                  )

    if not os.path.isfile(conf_path):
        raise utils.FileNotFoundError(f"No such file: {conf_path}")

    # Check data path:
    if os.path.splitext(data_path)[1] not in ['.csv', '.xlsx']:
        raise utils.FiletypeError(
                                  f"Data file does not end in .csv"
                                  f"or .xlsx: '{data_path}'"
                                  )

    if not os.path.isfile(data_path):
        raise utils.FileNotFoundError(f"No such file: {data_path}")

    # Check output directory:

    if os.path.exists(outdir):
        try:
            os.rmdir(outdir)  # succeeds if empty
        except OSError:  # directory not empty
            log.warning(f"{outdir} not empty. Renaming...")
            now = datetime.now()
            outdir = outdir.rstrip(os.sep)  # remove trailing slash
            outdir = f"{outdir}_{now.month:02d}_{now.day:02d}" \
                     f"_{now.hour:02d}_{now.minute:02d}_{now.second:02d}"
    os.makedirs(outdir)
    log.info(f"Saving to directory '{outdir}'")

    return conf_path, data_path, outdir
