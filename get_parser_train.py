# Create command line application
import argparse


# Initiate the parser
parser = argparse.ArgumentParser(description="train your own CNN to identify flowers")


# Define the command-line arguments
parser.add_argument("-a", "--arch", 
                    choices=['vgg16', 'vgg19'],
                    default='vgg16',
                    help="define the CNN architecture")

parser.add_argument("-e", "--epochs",
                    type=int,
                    default=3,
                    help="set the number of training epochs")

parser.add_argument("-g", "--gpu",
                    choices=['on', 'off'],
                    default='on',
                    help="turn graphics processor on or off")

parser.add_argument("-l", "--learningrate", 
                    type=float,
                    default=0.001,
                    help="set the learning rate")

parser.add_argument("-u", "--hiddenunits",
                    type=int,
                    default=512,
                    help="set the number of hidden units")


# Parse the arguments
args = parser.parse_args()