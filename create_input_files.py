import argparse
from utils import create_input_files


def parse_args():
    parser = argparse.ArgumentParser(description="TODO")

    parser.add_argument(
        "--karpathy_json", default="data/caption_datasets/dataset_coco.json",
        help="JSON containing Karpathy splits")
    parser.add_argument("--output_dir", required=True, help="output_directory")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path=args.karpathy_json,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=args.output_dir,
                       max_len=50)
