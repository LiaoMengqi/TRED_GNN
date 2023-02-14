import base_model
from utils import EnhancedDict,gpu_setting
import json

def main():
    dataset = 'ICEWS14s'
    path = './data/' + dataset + '/'

    opts = EnhancedDict()
    opts.path = path
    opts.lr = 0.00005
    opts.hidden_dim = 64
    opts.attention_dim = 8
    opts.n_layer = 6
    opts.batch_size = 100
    opts.act = 'idd'
    opts.lamb = 0.00012
    opts.dropout = 0.25
    opts.epochs = 100
    opts.disable_bar = False
    
    stop_step = 5
    decline_step = 0
    
    # 自动选择合适的GPU
    gpu_setting()
    
    trainer = base_model.Trainer(opts)
    for epoch in range(opts.epochs):
        trainer.train_epoch()
        """if epoch > 0:
            if model.train_history[-1][1] < model.train_history[-2][1]:
                decline_step = decline_step + 1
            else:
                decline_step = 0
            if decline_step >= stop_step:
                print('best : mrr ',model.train_history[-stop_step][1],' hist@1 ',model.train_history[-stop_step][2],' hist@10 ',model.train_history[-stop_step][3])
                break"""

    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import sys
        import traceback
        traceback.print_exc(file=open("error_log.txt","a"))
        sys.exit(1)