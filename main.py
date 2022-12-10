import base_model


class Options(object):
    pass


if __name__ == '__main__':
    dataset = 'ICEWS14s'
    path = './data/' + dataset + '/'

    opts = Options
    opts.path = path
    opts.lr = 0.001
    opts.hidden_dim = 64
    opts.attention_dim = 5
    opts.n_layer = 5
    opts.batch_size = 10
    opts.act = 'idd'
    opts.lamb = 0.000132
    epochs = 10
    model = base_model.BaseModel(opts)
    for epoch in range(epochs):
        print("epoch ", epoch + 1, ":")
        model.train()
