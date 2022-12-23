import base_model


class Options(object):
    pass


if __name__ == '__main__':
    dataset = 'ICEWS14s'
    path = './data/' + dataset + '/'

    opts = Options
    opts.path = path
    opts.lr = 0.00005
    opts.hidden_dim = 64
    opts.attention_dim = 8
    opts.n_layer = 6
    opts.batch_size = 100
    opts.act = 'idd'
    opts.lamb = 0.00012
    opts.dropout = 0.25
    epochs = 100
    stop_step = 5
    decline_step = 0
    model = base_model.BaseModel(opts)
    for epoch in range(epochs):
        print("epoch ", epoch + 1, ":")
        model.train()
        if epoch > 0:
            if model.train_history[-1][1] < model.train_history[-2][1]:
                decline_step = decline_step + 1
            else:
                decline_step = 0
            if decline_step >= stop_step:
                print('best : mrr ',model.train_history[-stop_step][1],' hist@1 ',model.train_history[-stop_step][2],' hist@10 ',model.train_history[-stop_step][3])
                break
