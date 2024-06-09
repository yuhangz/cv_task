class LrScheduler:
    StepLR = 'StepLR'
    ExponentialLR = 'ExponentialLR'

class ActivationFunction:
    ReLU = 'ReLU'
    LeakyReLU = 'LeakyReLU'
    ELU = 'ELU'

class OptimizerType:
    SGD = 'SGD'
    Adam = 'Adam'
    RMSprop = 'RMSprop'
    Adagrad = 'Adagrad'
    ASGD = 'ASGD'

class Regularization:
    L1 = 'L1'
    L2 = 'L2'
    none = None

class HyperParameter:
    lr_scheduler: LrScheduler
    activation_function: ActivationFunction
    optimizer_type: OptimizerType
    use_batch_normalization: bool
    regularization: Regularization
    dropout_rate: float

is_normal = lambda x: not x.startswith('__')
get_ns = lambda cls : list(filter(is_normal, dir(cls)))

search_space = {
    'lr_scheduler': get_ns(LrScheduler),
    'activation_function': get_ns(ActivationFunction),
    'optimizer_type': get_ns(OptimizerType),
    'use_batch_normalization': [True, False],
    'regularization': get_ns(Regularization),
    'dropout_rate': [0, 0.1, 0.5]
}

for k in search_space:
    print(k)