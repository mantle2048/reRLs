# -*- coding: utf-8 -*-
# %matplotlib notebook
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
from reRLs.infrastructure import logger_rlkit as log
from reRLs.infrastructure import tabulate as tab
from reRLs import user_config as conf
import numpy as np

pter = log.TerminalTablePrinter() #Print Table in Terminalj
tab_data = [['AverageReward', 1000], ['Q Value', 123.456]]

pter.headers = ['Exp Name', 'HalfCheetah-Td3']

# 初始化Logger 目录中包含空白的 progress.csv 和 debug.log
logger = log.setup_logger(exp_prefix='HalfCheetah_TD3')

logger.log("Hello World !")

logger.log("Hello World !", with_timestamp=False)

# 个性化定制种子和实验id
log.create_log_dir(exp_prefix='HalfCheetah_TD3', base_log_dir=conf.LOCAL_LOG_DIR, seed=1, exp_id=10)

# 不包含实验前缀的子目录
log.create_log_dir(exp_prefix='HalfCheetah_TD3', base_log_dir=conf.LOCAL_LOG_DIR, seed=2, exp_id=0, include_exp_prefix_sub_dir=False)

# 正常log，会打印 时间戳 + log目录 + log信息 的字符串
logger.log("123")

# 不带时间戳的 log
logger.log("123", with_timestamp=False)

# 不带 log目录 的字符串
logger.log("123", with_prefix=False)

logger.record_dict(dict(Average_Reward=1, Q_values=100))
logger.dump_tabular()
logger.record_dict(dict(Average_Reward=2, Q_values=200))
logger.dump_tabular()
logger.record_dict(dict(Average_Reward=3, Q_values=300))
logger.dump_tabular()

logger.record_dict(dict(Average_Reward=1, Q_values=100), prefix='123')
logger.dump_tabular()

logger.record_dict(dict(Average_Reward=1, Q_values=100), prefix='123')
logger.dump_tabular()

logger.record_dict(dict(Average_Reward=1, Q_values=100), prefix='456')
logger.dump_tabular()

logger._tabular_prefixes

# 初始化Logger 目录中包含空白的 progress.csv 和 debug.log
logger = log.setup_logger(exp_prefix='HalfCheetah_TD3')

logger.record_dict(dict(Env='HalfCheetah', Date=100), prefix='_')
logger.record_tabular_misc_stat("Average_Reward", np.arange(1, 100))
logger.dump_tabular()

# 初始化Logger 目录中包含空白的 progress.csv 和 debug.log
logger = log.setup_logger(exp_prefix='HalfCheetah_TD3')

# 初始化一个variant.json 可以用来记录某些重要参数发生的变化
logger = log.setup_logger(exp_prefix='HalfCheetah_TD3', variant={'a':1, 'b':2})

# snapshot_mode 用于确认储存参数时的频率 可选模式为[ all: 全部记录; last: 只记录最后一次; gap: 每隔snapshot_gap 次记录一次， gap_and_last: 顾名思义]
logger = log.setup_logger(exp_prefix='HalfCheetah_TD3', snapshot_mode='gap', snapshot_gap=1)

# 只输出tabulate到命令行， 不记录debug.log文件, 设置 log_tabular_only = True
logger = log.setup_logger(exp_prefix='HalfCheetah_TD3')
logger.record_dict(dict(Env='HalfCheetah', Date=100), prefix='_')
logger.record_tabular_misc_stat("Reward", np.arange(1, 100))
logger.dump_tabular()

logger = log.setup_logger(exp_prefix='HalfCheetah_TD3')
logger.record_dict(dict(Env='HalfCheetah', Date=100), prefix='_')
logger.record_tabular_misc_stat("Reward", np.arange(1, 100))

logger._prefixes

logger._prefix_str

logger._tabular_prefixes

logger._tabular_prefixes

logger._tabular

logger._text_outputs

logger._tabular_outputs

logger._text_fds

logger._tabular_fds

logger._tabular_header_written

logger._snapshot_dir

logger._snapshot_mode

logger._snapshot_gap

logger._header_printed

logger._log_tabular_only

logger.table_printer.headers

logger.table_printer.tabulars

import os
from collections import OrderedDict
log_file = os.path.join(logger._snapshot_dir, 'test.json')
dic = OrderedDict({'step':100000, 'learning_rate':'0.01->0.001'})
print(dic)
logger.log_variant(log_file, dic)

logger = log.setup_logger(exp_prefix='HalfCheetah_TD3')

# 上下文管理器修改 logger的前缀
with logger.prefix('* '):
    logger.record_dict(dict(Env=123, Date=100), prefix='_')
    logger.record_tabular_misc_stat("Reward", np.arange(1, 100))
    logger.dump_tabular()

# 上下文管理器修改 tabular 前缀
with logger.tabular_prefix('* '):
    logger.record_dict(dict(Env=123, Date=100), prefix='_')
    logger.record_tabular_misc_stat("Reward", np.arange(1, 100))
    logger.dump_tabular()

logger.record_dict(dict(Env=123, Date=100), prefix='_')
logger.record_tabular_misc_stat("Reward", np.arange(1, 100))
logger.dump_tabular()
