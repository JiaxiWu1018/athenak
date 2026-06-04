# A simple script for converting a collection of .bin files to .athdf/.xdmf files using
# bin_convert

# Python modules
import os
import argparse
import glob

# AthenaK modules
import bin_convert


def outputs_exist(fname):
    athdf_name = fname.replace(".bin", ".athdf")
    xdmf_name = athdf_name + ".xdmf"
    return athdf_name, xdmf_name, os.path.exists(athdf_name) and os.path.exists(xdmf_name)


# Main function
def main(**kwargs):
    # Get the root name for the file.
    files = sorted(glob.glob(kwargs['file_stem'] + '*.bin'))
    if len(files) < 1:
        print(f"No files found with stem {kwargs['file_stem']}")
        quit()

    if kwargs['skip_existing']:
        pending_files = []
        skipped = 0
        for fname in files:
            _, _, converted = outputs_exist(fname)
            if converted:
                skipped += 1
                if kwargs['verbose']:
                    print(f"Skipping existing outputs for {fname}")
                continue
            pending_files.append(fname)
        files = pending_files
        if kwargs['verbose']:
            print(f"Skipping {skipped} files with existing .athdf/.xdmf outputs")
        if len(files) < 1:
            print("All matching .bin files have already been converted.")
            return

    total = len(files)
    count = 1

    for fname in files:
        if kwargs['verbose']:
            print(f'Converting {count}/{total}: {fname}')
        athdf_name, xdmf_name, _ = outputs_exist(fname)
        filedata = bin_convert.read_binary(fname)
        bin_convert.write_athdf(athdf_name, filedata)
        bin_convert.write_xdmf_for(xdmf_name, os.path.basename(athdf_name), filedata)
        count = count+1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_stem', help='path to files, excluding .#.bin')
    parser.add_argument('--skip-existing', action='store_true',
                        help='only convert bins whose .athdf or .xdmf outputs are missing')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print file conversion progress')
    args = parser.parse_args()
    main(**vars(args))
