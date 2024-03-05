from ..model import TMC
from utils import *
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)
    for i, (data, target) in enumerate(tqdm_gen, 1):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())
        target = Variable(target.long().cuda())

        # refresh the optimizer
        optimizer.zero_grad()
        evidences, evidences_a, loss = model(data, target, epoch)  # 其中evidences_a中存放的就是分类logits
        acc = count_acc(logits=evidences_a, label=target)  # 分类准确率

        total_loss = loss
        
        lrc = scheduler.get_last_lr()[0]
        print(loss.item())
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, loss.item(), acc)
        )
        tl.add(loss.item())
        ta.add(acc)

        loss.backward()
        optimizer.step()
     
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way  # 测试用的类
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for batch_idx, (data, target) in enumerate(tqdm_gen, 1):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            target = Variable(target.long().cuda())
            evidences, evidence_a, loss = model(data, target, epoch)
            acc = count_acc(evidence_a, target)
            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va

def replace_base_fc(trainset, model, args):
    model = model.eval()

    trainloader = DataLoader(dataset=trainset, batch_size=64, num_workers=8, pin_memory=True, shuffle=False)
    embedding_list = []
    label_list = []

    with torch.no_grad():
        for i, (data, label) in enumerate(trainloader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            label = Variable(label.long().cuda())
            model.module.mode = 'encoder'
            evidences, evidence_a, loss = model(data, label, args.epochs_base - 1)
            embedding = evidence_a
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())

    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    # 这一部分还不太确定，有待调试
    for classifier in model.module.Classifiers:
        for fc_layer in classifier.fc[:-1]:
            if fc_layer.weight.data.size(0) >= args.base_class:
                 fc_layer.weight.data[:args.base_class] = proto_list

    return model