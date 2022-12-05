from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/Users/harddrive/Downloads/caption data/dataset_coco.json',
                       image_folder='/Users/harddrive/Downloads/coco_src',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/Users/harddrive/Downloads/coco',
                       max_len=50)
