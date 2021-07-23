from torchvision import transforms, datasets
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def evaluteTopK(k, model, loader):
    model.eval()

    class_correct = [0. for i in range(11)]
    class_total = [0. for i in range(11)]

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            y_resize = labels.view(-1, 1)
            _, predicted = outputs.topk(k, 1, True, True)

            for i in range(len(predicted)):
                class_total[labels[i]] += 1
                #print(torch.eq(predicted[i], y_resize[i]).sum().float().item())
                class_correct[labels[i]] += torch.eq(predicted[i], y_resize[i]).sum().float().item()

    for i in range(11):
        print('Top %d Accuracy of class %2d is %3d/%3d  %.2f%%' % (
            k, i, class_correct[i], class_total[i], (100 * class_correct[i] / class_total[i])))

    print('Top %d accuracy of the network on the %d test images: %d/%d  %.2f %%'
          % (k, sum(class_total), sum(class_correct), sum(class_total), (100 * sum(class_correct) / sum(class_total))))

    return 100 * sum(class_correct) / sum(class_total)


if __name__ == '__main__':

    transform_train = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    evalset = datasets.ImageFolder(root='food11re/evaluation/', transform=transform_train)
    evalloader = DataLoader(dataset=evalset, batch_size=20, num_workers=3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    choice = input('Evaluation: 1.teacher 2.student ===> ')

    if choice == '1':
        net = torchvision.models.resnet18(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, 11)
        net.load_state_dict(torch.load('teacher_model.pth'))
    elif choice == '2':
        net = torch.load('student_model.pth', map_location='cuda:0')
    else:
        print('Choice error.')
        exit()

    net.to(device)
    net.eval()

    print(evaluteTopK(1, net, evalloader))
    print(evaluteTopK(3, net, evalloader))

