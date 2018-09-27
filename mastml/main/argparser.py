__all__ = ['get_commandline_args']

import argparse
import os


def get_commandline_args():
    parser = argparse.ArgumentParser(description=(
                                                  'MAterials Science '
                                                  'Toolkit - Machine Learning'
                                                  ))

    parser.add_argument(
                        'conf_path',
                        type=str,
                        help='path to mastml .conf file'
                        )

    parser.add_argument(
                        'data_path',
                        type=str,
                        help='path to csv or xlsx file'
                        )

    parser.add_argument(
                        '-o',
                        action="store",
                        dest='outdir',
                        default='results',
                        help=(
                              'Folder path to save output '
                              'files to. Defaults to results/'
                              )
                        )

    # from https://stackoverflow.com/a/14763540
    # we only use them to set a bool but it
    # would be nice to have multiple levels in the future
    parser.add_argument(
                        '-v',
                        '--verbosity',
                        action="count",
                        help="include this flag for more verbose output"
                        )

    parser.add_argument(
                        '-q',
                        '--quietness',
                        action="count",
                        help=(
                              "include this flag to hide [DEBUG]"
                              "printouts, or twice to hide [INFO]"
                              )
                        )

    args = parser.parse_args()

    verbosity = (
                 (args.verbosity if args.verbosity else 0) -
                 (args.quietness if args.quietness else 0)
                 )

    # verbosity -= 1 ## uncomment this for distribution
    return (
            os.path.abspath(args.conf_path),
            os.path.abspath(args.data_path),
            os.path.abspath(args.outdir),
            verbosity
            )
