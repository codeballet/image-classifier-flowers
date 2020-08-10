import argparse


# Initiate the parser
parser = argparse.ArgumentParser(description="get the class probability of an image using a checkpoint")


# Define the command-line arguments
parser.add_argument('-c', '--checkpoint',
                    default='checkpoint.pth',
                    help='state checkpoint filepath')

parser.add_argument('-f', '--flower',
                    default='flowers/test/1/image_06743.jpg',
                    help='state flower filepath')

parser.add_argument('-g', '--gpu',
                    choices=['on', 'off'],
                    default='on',
                    help='turn graphics processor on or off')

parser.add_argument('-k',
                    type=int,
                    default=1,
                    help='define the amount of classes and probabilities to print')

parser.add_argument('-n', '--names',
                    default='cat_to_name.json',
                    help='json file path with category names')

# Parse the arguments
args = parser.parse_args()