from reRLs.infrastructure.loggers import setup_logger
# %matplotlib notebook
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

from aim import Run
run = Run()

run.name = '123'

hparams_dict = {
    'learning_rate': 0.001,
    'batch_size': 32,
}
run['hparams'] = hparams_dict

run.track(3.0, name='loss')

# log metric
for i in range(10):
    run.track(i, name='numbers')








