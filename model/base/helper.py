from ..model import TMC
from utils import *
from tqdm import tqdm
from torch.autograd import Variable

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)

    for i, (data, target) in enumerate(tqdm_gen, 1):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num])