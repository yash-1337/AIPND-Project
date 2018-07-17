import torch
import json

import argparse
from helper import *

def get_input_args():

    parser = argparse.ArgumentParser(description = 'Parameter Options for Prediction Using the Neural Network')
    
    parser.add_argument('image_path', action = 'store')
    parser.add_argument('checkpoint_path', action = 'store')
    parser.add_argument('--top_k', action = 'store', type = int, dest = 'top_k', default = 5)
    parser.add_argument('--category_names', action = 'store', type = str, dest = 'category_names')
    parser.add_argument('--gpu', action = 'store_true', dest = 'gpu')
    
    return parser.parse_args()


def main():
    input_args = get_input_args()
    
    image_path = input_args.image_path
    checkpoint_path = input_args.checkpoint_path
    top_k = input_args.top_k
    cat_to_name_path = input_args.category_names
    gpu_is_enabled = input_args.gpu

    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu_is_enabled else "cpu")
    
    model = LoadCheckpoint(checkpoint_path)
    
    probs, classes = Predict(image_path, model, device, top_k)
    
    if cat_to_name_path is None:
        ViewPredictionResults(probs, classes)
    else:
        with open(cat_to_name_path, 'r') as f:
            cat_to_name = json.load(f)
            
        named_classes = [cat_to_name[i] for i in classes]
        
        ViewPredictionResults(probs, named_classes)
    
    
if __name__ == '__main__':
    main()

