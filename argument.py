
class Argument:

    def __init__(self):
        self.hidden_dim = 500
        self.lambda_value = 10
        self.train_epoch = 200
        self.batch_size = 100
        self.optimizer_method = 'Adam'
        self.base_lr = 1e-3
        self.decay_epoch_step = 10
        self.random_seed = 1000
        self.display_step = 1
        self.ratio = 0.9
        self.repeat_number = 1

