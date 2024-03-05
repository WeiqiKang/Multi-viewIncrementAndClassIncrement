from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
from model.data import *
from model.model import TMC
import torch
import time
from .helper import *

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        self.set_save_path()

    def set_up_model(self): # 设置当前训练类的model
        args = self.args
        self.model = TMC(args.num_classes, args.views, args.dims, args.lambda_epochs)
        self.model = nn.DataParallel(self.model, list(range(args.num_gpu)))
        self.model = self.model.cuda()

        if args.model_dir != None:
            print('Loading init parameters from %s' % args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print("random init params")
            if args.start_session > 0:
                print("WARNING: Random init weights for new sessions!")
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_dataloader(self, session, train_set_by_class, test_set):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args, train_set_by_class, test_set)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session, train_set_by_class, test_set)

        return trainset, trainloader, testloader

    def get_optimizer_base(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler
    
    def train(self):
        args = self.args
        t_start_time = time.time()

        result_list = [args]
        
        # 完整的数据集
        self.full_dataset = Multi_view_data(args.data_path)

        for session in range(args.start_session, args.sessions):
            # 分割数据集，按类别分割，保存在数组里
            train_set_by_class, test_set = split_dataset_by_class(self.full_dataset, session, args)
            # 上述函数已经取出了包含args.base_class + session * args.way这些类的数据集，按类别进行索引
            train_set, trainloader, testloader = self.get_dataloader(session, train_set_by_class, test_set)
 
            self.model.load_state_dict(self.best_model_dict) 

            if session == 0:
                print('new classes for this session:\n', list(range(60)))
                optimizer, scheduler = self.get_optimizer_base()
                
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # 在上面取出的base数据集上训练
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    # 在所有seen class上进行测试
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))
                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)

                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()
                
                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    tsl, tsa = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
            else:
                print("training session: [%d]" % session)
                self.model.eval()
                start_class = args.base_class + args.way * (session - 1)
                end_class = start_class + args.way - 1
                class_list = np.array(list(range(start_class, end_class + 1)))
                self.model.module.update_fc(trainloader, class_list, session)

                tsl, tsa = test(self.model, testloader, 0, args, session)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        self.args.save_path = self.args.save_path + '-start_%d/' % (self.args.start_session)

        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
            
        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)

        ensure_path(self.args.save_path)
        return None