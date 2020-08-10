import torch
from torch import optim

from workspace_utils import active_session
from get_parser_train import args
from get_classifier import get_classifier
from get_device import get_device
from get_data import image_datasets, dataloaders
from get_model import get_model
from get_checkpoints import save_checkpoint


# Create, train and save a model as a checkpoint
def main():
    # Define variables
    step = 0
    running_loss = 0
    print_every = 5
    epochs = args.epochs
    device = get_device(args.gpu)
    print(f'Device used: {device}')
    
    # Create model
    model = get_model(args.arch)
    criterion, model.classifier = get_classifier(args.hiddenunits)
    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learningrate)

    # Keep workspace active
    with active_session():
        # do long-running work here
        for epoch in range(epochs):
            print('Starting new epoch...')
            for images, labels in dataloaders['train']:
                model.train()
                step += 1

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                # Feed forward
                logps = model.forward(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Check progress using test_data
                if step % print_every == 0:
                    model.eval()
                    test_loss = 0
                    accuracy = 0

                    for images, labels in dataloaders['test']:
                        images, labels = images.to(device), labels.to(device)

                        logps = model.forward(images)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                    # Print results
                    print(f'Step {step}; '
                          f'Epoch {epoch+1}/{epochs}; '
                          f'Train loss {running_loss/print_every:.3f}; '
                          f'Test loss {test_loss/len(dataloaders["test"]):.3f}; '
                          f'Test accuracy: {accuracy/len(dataloaders["test"]):.3f}')

                    # Reset loss counter
                    running_loss = 0
        
        # Save model as a checkpoint
        save_checkpoint(model.classifier,
                        args.arch,
                        args.learningrate,
                        model.state_dict(),
                        image_datasets['train'].class_to_idx,
                        optimizer.state_dict(),
                        criterion)


if __name__ == "__main__":
    main()