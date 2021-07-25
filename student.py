import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional
import torch.utils.data
from torchsampler import ImbalancedDatasetSampler
from imgAugTransform import ImgAugTransform
import PIL

if __name__ == '__main__':
    print("==> Check devices..")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Current device: ", device)

    # Also can print your current GPU id, and the number of GPUs you can use.
    print("Our selected device: ", torch.cuda.current_device())
    print(torch.cuda.device_count(), " GPUs is available")

    print('==> Preparing dataset..')

    # The output of torchvision datasets are PILImage images of range [0, 1]
    # We transform them to Tensor type
    # And normalize the data
    # Be sure you do same normalization for your train and test data

    # The transform function for train data
    transform_train = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # The transform function for validation and evaluation data
    transform_test = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Use API to load train dataset
    trainset = torchvision.datasets.ImageFolder(root='food11re/training', transform=transform_train)

    # Use API to load valid dataset
    validset = torchvision.datasets.ImageFolder(root='food11re/validation', transform=transform_test)

    # randon_sampler = torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=20000)
    imblanceSampler = ImbalancedDatasetSampler(trainset, num_samples=18000)

    # Create DataLoader to draw samples from the dataset
    # In this case, we define a DataLoader to random sample our dataset.
    # For single sampling, we take one batch of data. Each batch consists 4 images
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                              shuffle=False, num_workers=0, sampler=imblanceSampler)

    validloader = torch.utils.data.DataLoader(validset, batch_size=20,
                                              shuffle=True, num_workers=2)

    checkpoint = 0
    print('Trying to load checkpoint ...')
    try:
        checkpoint = torch.load('student_checkpoint.tar')
    except:
        print('Checkpoint not found.')

    print('==> Building model..')

    # declare a new model
    # net = torchvision.models.mobilenet_v3_small(pretrained=True)
    net = torchvision.models.mobilenet_v3_small(pretrained=True)

    net.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=1024, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1024, out_features=11, bias=True)
    )

    if checkpoint != 0:
        net.load_state_dict(checkpoint['net_state_dict'])

    net = net.to(device)
    print(net)

    print('==> Defining loss function and optimize..')
    import torch.optim as optim

    # loss function
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.KLDivLoss()

    # optimization algorithm
    optimizer = optim.Adam(net.parameters())
    if checkpoint != 0:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load Teacher model
    teacherNet = torchvision.models.resnet18(pretrained=False)
    num_features = teacherNet.fc.in_features
    teacherNet.fc = nn.Linear(num_features, 11)
    teacherNet.load_state_dict(torch.load('teacher_model.pth'))
    teacherNet = teacherNet.to(device)

    print('==> Training model..')
    # Set the model in training mode
    # because some function like: dropout, batchnorm...etc, will have
    # different behaviors in training/evaluation mode
    # [document]: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train

    # number of epochs to train the model
    n_epochs = 40

    valid_loss_min = np.Inf if checkpoint == 0 else checkpoint['valid_loss']  # track change in validation loss

    alpha = 0.8

    for epoch in range(1 if checkpoint == 0 else checkpoint['epoch'], n_epochs + 1):

        # keep track of training and validation loss
        train_correct = 0
        valid_correct = 0
        train_loss = 0.0
        valid_loss = 0.0
        print('running epoch: {}'.format(epoch))
        ###################
        # train the model #
        ###################
        net.train()
        teacherNet.eval()
        for data, target in tqdm(trainloader):
            # move tensors to GPU if CUDA is available

            data, target = data.cuda(), target.cuda()

            soft_target = teacherNet(data)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = net(data)

            # select the class with highest probability
            _, pred = output.max(1)

            # if the model predicts the same results as the true
            # label, then the correct counter will plus 1
            train_correct += pred.eq(target).sum().item()

            # calculate the batch loss
            loss1 = criterion(output, target)

            T = 2
            outputs_S = nn.functional.log_softmax(output / T, dim=1)
            outputs_T = nn.functional.softmax(soft_target / T, dim=1)
            loss2 = criterion2(outputs_S, outputs_T) * T * T
            loss = loss1 * (1 - alpha) + loss2 * alpha

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        net.eval()
        with torch.no_grad():
            for data, target in tqdm(validloader):
                # move tensors to GPU if CUDA is available

                data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = net(data)

                # select the class with highest probability
                _, pred = output.max(1)

                # if the model predicts the same results as the true
                # label, then the correct counter will plus 1
                valid_correct += pred.eq(target).sum().item()

                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)

        # calculate average losses
        # train_losses.append(train_loss/len(train_loader.dataset))
        # valid_losses.append(valid_loss.item()/len(valid_loader.dataset)
        train_loss = train_loss / len(trainloader.sampler)
        valid_loss = valid_loss / len(validloader.dataset)
        train_correct = 100. * train_correct / len(trainloader.sampler)
        valid_correct = 100. * valid_correct / len(validset)

        # print training/validation statistics
        print(
            '\tTraining Acc: {:.6f} \tTraining Loss: {:.6f} \tValidation Acc: {:.6f} \tValidation Loss: {:.6f}'.format(
                train_correct,
                train_loss, valid_correct, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving checkpoint ...'.format(
                valid_loss_min,
                valid_loss))

            torch.save({
                'epoch': epoch + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'net_state_dict': net.state_dict()
            }, 'student_checkpoint.tar')

            valid_loss_min = valid_loss

    print('Saving model ...')
    checkpoint = torch.load('student_checkpoint.tar')
    net.load_state_dict(checkpoint['net_state_dict'])
    torch.save(net, 'student.pth')
    print('Finished Training')

