from get_parser_predict import args
from get_checkpoints import load_checkpoint
from get_prediction import predict


def main():
    # Load a checkpoint
    optimizer, criterion, model = load_checkpoint(args.checkpoint)
    
    # Make prediction
    probabilities, labels, flowers = predict(args.flower, model, args.names, args.k)
    
    print(f'Flowers: {flowers}',
          f'Probabilities: {probabilities}',
          f'Labels: {labels}')
    

if __name__ == "__main__":
    main()