from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr30k',
                       karpathy_json_path='./caption_datasets/dataset_flickr30k.json',
                       image_folder='./flickr30k_images/flickr30k_images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='./flickr30k_output',
                       max_len=20)
