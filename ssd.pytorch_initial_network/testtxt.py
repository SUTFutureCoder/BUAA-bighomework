import torch
from torch.utils.data import DataLoader
from loader import collater, Resizer, AspectRatioBasedSampler, Normalizer,TXTDataset
from torchvision import transforms
import argparse
import tools

ap     = argparse.ArgumentParser(description='Just for eval.')
ap.add_argument('-source','--source')
ap.add_argument('-anno','--anno')
ap.add_argument('-image','--image')
ap.add_argument('-model','--model')
args = vars(ap.parse_args())

dataset_val = TXTDataset(source=args['source'], anno=args['anno'], image =args['image'], transform=transforms.Compose([Normalizer(), Resizer()]))
sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
modelretina = torch.load(args['model'])
modelretina.cuda()
modelretina.eval()
AP = tools.evaluate(dataset_val, modelretina)
map = (AP[0][0]+AP[1][0])/2
print("mAp:"+str(map))
