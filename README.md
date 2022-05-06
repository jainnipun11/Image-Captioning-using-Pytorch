# Image-Captioning-using-Pytorch

## Objective

To generate accurate image captions for Remote Sensing Images.


<img width="446" alt="output1" src="https://user-images.githubusercontent.com/96676539/167213754-27ff94df-2492-4e2b-ac46-4d4a01a9fe20.png">

## Implementation
The sections below briefly describe the implementation.

They are meant to provide some context, but details are best understood directly from the code, which is quite heavily commented.

### Inputs to model
We will need three inputs.

### Images
Since we're using a pretrained Encoder, we would need to process the images into the form this pretrained Encoder is accustomed to.

Pretrained ImageNet models available as part of PyTorch's torchvision module. This page details the preprocessing or transformation we need to perform – pixel values must be in the range [0,1] and we must then normalize the image by the mean and standard deviation of the ImageNet images' RGB channels.

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
Also, PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions.

### Captions
Captions are both the target and the inputs of the Decoder as each word is used to generate the next word.

To generate the first word, however, we need a zeroth word, <start>.

At the last word, we should predict <end> the Decoder must learn to predict the end of a caption. This is necessary because we need to know when to stop decoding during inference.

<start> a man holds a football <end>

Since we pass the captions around as fixed size Tensors, we need to pad captions (which are naturally of varying length) to the same length with <pad> tokens.

<start> a man holds a football <end> <pad> <pad> <pad>....

Furthermore, we create a word_map which is an index mapping for each word in the corpus, including the <start>,<end>, and <pad> tokens. PyTorch, like other libraries, needs words encoded as indices to look up embeddings for them or to identify their place in the predicted word scores.

9876 1 5 120 1 5406 9877 9878 9878 9878....

Therefore, captions fed to the model must be an Int tensor of dimension N, L where L is the padded length.
  
  ![image](https://user-images.githubusercontent.com/96676539/167215792-41bd4101-a7ae-44f6-b750-673277c61e28.png) 
  
  
### Data pipeline
  
See create_input_files() in utils.py.

This reads the data downloaded and saves the following files –

An HDF5 file containing images for each split in an _*I, 3, 256, 256*_ tensor, where I is the number of images in the split. Pixel values are still in the range [0, 255], and are stored as unsigned 8-bit Ints.
  
A JSON file for each split with a list of N_c * I encoded captions, where N_c is the number of captions sampled per image. These captions are in the same order as the images in the HDF5 file. Therefore, the ith caption will correspond to the i // N_cth image.
  
A JSON file for each split with a list of N_c * I caption lengths. The ith value is the length of the ith caption, which corresponds to the i // N_cth image.
  
A JSON file which contains the word_map, the word-to-index dictionary.
  
Before we save these files, we have the option to only use captions that are shorter than a threshold, and to bin less frequent words into an <unk> token.

We use HDF5 files for the images because we will read them directly from disk during training / validation. They're simply too large to fit into RAM all at once. But we do load all captions and their lengths into memory.
  
## Beam Search
  
We use a linear layer to transform the Decoder's output into a score for each word in the vocabulary.

The straightforward – and greedy – option would be to choose the word with the highest score and use it to predict the next word. But this is not optimal because the rest of the sequence hinges on that first word you choose. If that choice isn't the best, everything that follows is sub-optimal. And it's not just the first word – each word in the sequence has consequences for the ones that succeed it.

It might very well happen that if you'd chosen the third best word at that first step, and the second best word at the second step, and so on... that would be the best sequence you could generate.

It would be best if we could somehow not decide until we've finished decoding completely, and choose the sequence that has the highest overall score from a basket of candidate sequences.

Beam Search does exactly this.

At the first decode step, consider the top k candidates.
Generate k second words for each of these k first words.
Choose the top k [first word, second word] combinations considering additive scores.
For each of these k second words, choose k third words, choose the top k [first word, second word, third word] combinations.
Repeat at each decode step.
After k sequences terminate, choose the sequence with the best overall score.

![image](https://user-images.githubusercontent.com/96676539/167216170-53dae1a1-cf68-49a0-81d8-934136be6ffa.png)


