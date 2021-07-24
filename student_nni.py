import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional
import torch.utils.data
from autoaugment import ImageNetPolicy
from torchsampler import ImbalancedDatasetSampler
from imgAugTransform import ImgAugTransform
import PIL
import logging
import nni
from nni.utils import merge_parameter

logger = logging.getLogger('student_nni_AutoML')

def main(args):

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                              shuffle=False, num_workers=0, sampler=imblanceSampler)

    validloader = torch.utils.data.DataLoader(validset, batch_size=args['batch_size'],
                                              shuffle=True, num_workers=2)

    print('==> Building model..')

    # declare a new model
    #net = torchvision.models.mobilenet_v3_small(pretrained=True)
    net = torchvision.models.mobilenet_v3_small(pretrained=True)

    net.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=1024, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1024, out_features=11, bias=True)
    )

    # net.fc = nn.Sequential(
    #    nn.Dropout(0.5),
    #    nn.Linear(num_features, 11)
    # )

    # change all model tensor into cuda type
    # something like weight & bias are the tensor
    net = net.to(device)
    print(net)
    '''
    for k, v in net.named_parameters():
        print(k)
        if (k == 'conv1.weight' or k == 'bn1.weight' or k == 'bn1.bias'):
            v.requires_grad = False
        if (k[0:6] == 'layer1' or k[0:6] == 'layer2'):
            v.requires_grad = False
    '''

    print('==> Defining loss function and optimize..')
    import torch.optim as optim

    # loss function
    if(args['criterion'] == 'CrossEntropyLoss'):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.KLDivLoss()

    if (args['criterion2'] == 'CrossEntropyLoss'):
        criterion2 = nn.CrossEntropyLoss()
    else:
        criterion2 = nn.KLDivLoss()

    # optimization algorithm
    optimizer = optim.Adam(net.parameters())

    # Load Teacher model
    teacherNet = torchvision.models.resnet18(pretrained=False)
    num_features = teacherNet.fc.in_features
    teacherNet.fc = nn.Linear(num_features, 11)
    teacherNet.load_state_dict(torch.load('teacher_model.pth'))
    teacherNet = teacherNet.to(device)

    print('==> Training model..')

    # number of epochs to train the model
    n_epochs = 40

    valid_loss_min = np.Inf  # track change in validation loss

    alpha = args['alpha']

    for epoch in range(1, n_epochs + 1):

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
        nni.report_intermediate_result(valid_correct)
        # print training/validation statistics
        print(
            '\tTraining Acc: {:.6f} \tTraining Loss: {:.6f} \tValidation Acc: {:.6f} \tValidation Loss: {:.6f}'.format(
                train_correct,
                train_loss, valid_correct, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(net, 'student_nni_model.pth')
            valid_loss_min = valid_loss

    nni.report_final_result(valid_correct)
    print('Finished Training')

if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        '''
        tuner_params = {
    "batch_size": 20,
    "criterion": "CrossEntropyLoss",
    "criterion2": "KLDivLoss",
    "alpha": 0.5
}
        '''
        logging.debug(tuner_params)
        main(tuner_params)
    except Exception as exception:
        raise