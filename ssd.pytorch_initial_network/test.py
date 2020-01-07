from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import SIXrayAnnotationTransform, SIXrayDetection, BaseTransform, SIXray_CLASSES
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/test.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./../predicted_file/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
#parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename_have = save_folder+'det_test_带电芯充电宝.txt'
    filename_not_have = save_folder+'det_test_不带电芯充电宝.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        img_name = testset.get_image_name(i)
        if img_name.count("less"):
            filename = filename_not_have
        else:
            filename = filename_have

        with open(filename, mode='a') as f:
            f.write(img_id+" ")
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(score).replace("tensor(", "").replace(")", "") + ' ' + ' '.join(str(c) for c in coords))
                j += 1
        with open(filename, mode='a') as f:
          f.write("\n")


def test_voc():
    # load net
    num_classes = len(SIXray_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model,map_location=torch.device('cpu')))
    net.eval()
    print('Finished loading model!')
    # load data
    test_sets = "./data/sixray/test_1650.txt"
    testset = SIXrayDetection(test_sets, None, SIXrayAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

def test(img_path, anno_path):
    # load net
    num_classes = len(SIXray_CLASSES) + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu')))
    net.eval()
    # read and put into a file
    test_sets = []
    for anno_file in os.listdir(anno_path):
        test_sets.append(anno_file.split('.')[0])
    testset = SIXrayDetection(test_sets, None, SIXrayAnnotationTransform(), image_path=img_path, anno_path=anno_path)
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)


if __name__ == '__main__':
    # test_voc()
    img_path = '/Users/baidu/Github/BUAA-bighomework/ssd.pytorch_initial_network/Image_test'
    anno_path = '/Users/baidu/Github/BUAA-bighomework/ssd.pytorch_initial_network/Anno_test'
    test(img_path, anno_path)
