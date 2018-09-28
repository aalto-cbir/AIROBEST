#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Training
"""
import argparse


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='Training options for hyperspectral data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-src_path',
                        required=False, type=str,
                        default='./data/hyperspectral_src.pt',
                        help='Path to training input file')
    parser.add_argument('-tgt_path',
                        required=False, type=str,
                        default='./data/hyperspectral_tgt.gt',
                        help='Path to training labels')

    # Training options
    train = parser.add_argument_group('Training')
    train.add_argument('-epoch', type=int,
                       default=10,
                       help="Number of training epochs, default is 10")
    train.add_argument('-patch_size', type=int,
                       default=11,
                       help="Size of the spatial neighbourhood, default is 11")
    train.add_argument('-lr', type=float,
                       default=1e-3,
                       help="Learning rate, default is 1e-3")
    train.add_argument('-batch_size', type=int,
                       default=64,
                       help="Batch size, default is 64")
    opt = parser.parse_args()

    return opt


def main():
    print('Start training...')
    #######
    options = parse_args()
    print('Training options: {}'.format(options))

    print('End training...')


if __name__ == "__main__":
    main()
