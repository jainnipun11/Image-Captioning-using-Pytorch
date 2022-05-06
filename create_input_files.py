from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='E:/IIT Study/Datasets/Sydney_captions/dataset.json',
                       image_folder='E:/IIT Study/Datasets/Sydney_captions/imgs',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='E:/IIT Study/Datasets/Sydney_captions/result',
                       max_len=50)
