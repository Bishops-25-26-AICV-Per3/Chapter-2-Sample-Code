import collections
import torch, torchvision

# Mac users only
torch.set_default_device(torch.device('mps'))

torch.manual_seed(37)

# Load the ResNet pre-trained weights
# Need to do this before loading data to get preprocess function
weights = torchvision.models.ResNet50_Weights.DEFAULT
res = torchvision.models.resnet50(weights = weights)

def get_dataloaders(
        batch_size: int, train_proportion: float,
) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """Set up train and validation datasets"""
    path_to_data = "defungi"
    # Load the preprocessing function
    transforms = torchvision.transforms.Compose([
            weights.transforms(),
    ])
    # Pytorch has a two-step process: 1. dataset has all images 2. 
    #   dataloader for each subset: train, validation.
    full_dataset = torchvision.datasets.ImageFolder(
            path_to_data,
            transform = transforms,
    )
    generator = torch.Generator(device="mps").manual_seed(37)
    train_set, validation_set = torch.utils.data.random_split(
            full_dataset,
            lengths = [train_proportion, 1-train_proportion],
            generator = generator,
    )
    train = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
    validation = torch.utils.data.DataLoader(validation_set, 
            batch_size = batch_size)
    return train, validation

train, validation = get_dataloaders(batch_size = 32, train_proportion = 0.8)

print(f"Number of training batches: {len(train)}")
print(f"Number of validation batches: {len(validation)}")

# Now we have data loaders ready.  Whoo-hoo!

# Look at res.children to get names of layers.  We see that last layer
#   is named fc.
# print(res.children)

# Get rid of last layer that classifies into 1000 classes.
res.fc = torch.nn.Identity()

model = torch.nn.Sequential(collections.OrderedDict([
    ('resnet', res),
    ('final', torch.nn.LazyLinear(5)),
    ('softmax', torch.nn.Softmax(dim = 1)),
]))

lr = 0.0001
param_groups = [
        {'params': model.resnet.parameters(), 'requires_grad': False},
        {'params': model.final.parameters(), 'lr': lr},
]
optimizer = torch.optim.Adam(param_groups)
# Set the model to training mode.
model.train()

loss_fn = torch.nn.CrossEntropyLoss()

# Number in the range here is # of epochs
for i in range(1):
    batch_losses = []
    batch_accuracies = []
    print(f"======= EPOCH {i+1} =======")
    for image_batch, label_batch in train:
        preds = model(image_batch)
        # In Pytorch, the loss object has a *ton* of information in it.
        loss = loss_fn(preds, label_batch)
        # Set up the network of calculations of loss to go backwards 
        #       so it can be trained.
        loss.backward()
        # Actually carry out that training.
        optimizer.step()
        # Reset the neural network for the next batch.
        optimizer.zero_grad()
        # The rest of this is just for output and record keeping
        # Use float on loss to just get the decimal value for the loss.
        batch_losses.append(float(loss))
        cur_loss = sum(batch_losses)/len(batch_losses)
        this_batch_acc = (int(sum(preds.argmax(1) == label_batch))/
                len(label_batch))
        batch_accuracies.append(this_batch_acc)
        cur_acc = sum(batch_accuracies)/len(batch_accuracies)
        print("Train:", end="\t\t")
        print(f"Batch: {len(batch_losses)}", end="\t")
        print(f"Loss: {round(cur_loss, 4)}", end="\t")
        print(f"Accuracy: {round(cur_acc, 4)}", end="\r")
    print()
    batch_losses = []
    batch_accuracies = []
    for image_batch, label_batch in validation:
        with torch.no_grad():
            preds = model(image_batch)
            loss = loss_fn(preds, label_batch)
            # Skip all the training steps here
            batch_losses.append(float(loss))
            cur_loss = sum(batch_losses)/len(batch_losses)
            this_batch_acc = (int(sum(preds.argmax(1) == label_batch))/
                    len(label_batch))
            batch_accuracies.append(this_batch_acc)
            cur_acc = sum(batch_accuracies)/len(batch_accuracies)
            print("Validation:", end="\t")
            print(f"Batch: {len(batch_losses)}", end="\t")
            print(f"Loss: {round(cur_loss, 4)}", end="\t")
            print(f"Accuracy: {round(cur_acc, 4)}", end="\r")
    print()