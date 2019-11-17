"""
main
"""

import os
import sys
import argparse
from application import Application


def parse_arguments(argv, prog=''):
    """ Parse command-line arguments. """
    # initialize the command-line parser
    parser = argparse.ArgumentParser(prog,
                                     description='naruto-vs main application')

    # add arguments
    parser.add_argument('--vid-file',
                        type=str,
                        help='Video file',
                        required=True)
    parser.add_argument('--enable-character-recognition',
                        type=bool,
                        help='Enable/disable character recognition',
                        default=True,
                        required=False)
    parser.add_argument('--enable-village-symbol-detection',
                        type=bool,
                        help='Enable/disable village symbol detection',
                        default=True,
                        required=False)
                        
    # run parser
    args, unprocessed_argv = parser.parse_known_args(argv)

    # return arguments
    return args, unprocessed_argv


def main(argv, prog=''):
    """ main """
    # get arguments from command-line
    args, unprocessed_argv = parse_arguments(argv, prog)

    # create and initialize application object
    app = Application()

    app.set_vidfile(args.vid_file)
    app.set_enable_character_recognition(args.enable_character_recognition)
    app.set_enable_village_symbol_detection(args.enable_village_symbol_detection)

    # run application
    app.run()

    return 0


if __name__ == '__main__':
    main(sys.argv[1:], sys.argv[0])
