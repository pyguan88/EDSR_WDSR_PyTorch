import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if __name__ == '__main__':
    if args.data_test == 'video':
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            model = model.Model(args, checkpoint)
            loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, model, loss, checkpoint)
            while not t.terminate():  # test_only则执行test，并跳过循环，否则循环epochs次
                t.train()  # train中不会保存model
                t.test()   # test中保存model
            checkpoint.done()

