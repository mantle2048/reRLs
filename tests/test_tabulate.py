# -*- coding: utf-8 -*-
# %matplotlib notebook
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
from reRLs.infrastructure import tabulate as tab

print(tab._pipe_segment_with_colons('left', 10))
print(tab._pipe_segment_with_colons('center', 10))
print(tab._pipe_segment_with_colons('right', 10))
print(tab._pipe_segment_with_colons('decimal', 10))

print(tab._pipe_line_with_colons([10, 20, 30, 40], ['left', 'center', 'right', 'decimal']))
print(tab._pipe_line_with_colons([20], ['center' ]))
print(tab._pipe_line_with_colons([30], ['right'  ]))
print(tab._pipe_line_with_colons([40], ['decimal']))

print(tab._mediawiki_row_with_attrs(separator = ' ///// ', cell_values = ['a', 'b', 'c'], colwidths = [100], colaligns = ['right', 'center', 'left']))

print(tab._latex_line_begin_tabular(colwidths = [231231313], colaligns = ['left']))

print(tab.simple_separated_format(separator='|'))

# +
# def tabulate(tabular_data, headers=[], tablefmt="simple", floatfmt="g", numalign="decimal", stralign="left", missingval=""):
# -

print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]])) # 默认按小数点对齐
print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]], numalign='left')) # 默认按小数点对齐
print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]], numalign='right')) # 右对齐
print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]], numalign='center')) # 中心对齐
print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]], numalign=None)) # 取消对齐

print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]], floatfmt="%")) # 百分数
print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]], floatfmt=",")) # 逗号分位
print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]], floatfmt=",")) # 逗号分位
print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]], floatfmt="e")) # 科学计数法
print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]], floatfmt="n")) # 浮点数时等于g
print(tab.tabulate([['first_row', 2.34], ['second_row', "8.999"], ["third_row", 10001]], floatfmt="g")) # 一般格式

data = dict(col1=[1,2], col2=[3,4], col3=[4,5])
print(tab.tabulate(tabular_data = data))

print(tabulate([["sex","age"],["Alice","F",24],["Bob","M",19]], headers="firstrow"))

data = dict(col1=[1,2], col2=[3,4], col3=[4,5])
print(tab.tabulate(tabular_data = data, headers='keys'))

data = dict(col1=[None, 1,2], col2=[3,4, None], col3=[4,None,5])
print(tab.tabulate(tabular_data = data, headers='keys', missingval="?"))

data = dict(col1=[1,2], col2=[3,4], col3=[4,5])
print(tab.tabulate(tabular_data = data, headers='keys', tablefmt='plain'))

data = dict(col1=[1,2], col2=[3,4], col3=[4,5])
print(tab.tabulate(tabular_data = data, headers='keys', tablefmt='simple'))

data = dict(col1=[1,2], col2=[3,4], col3=[4,5])
print(tab.tabulate(tabular_data = data, headers='keys', tablefmt='grid'))

data = dict(col1=[1,2], col2=[3,4], col3=[4,5])
print(tab.tabulate(tabular_data = data, headers='keys', tablefmt='pipe'))

data = dict(col1=['AverageReward','Q Values'], col2=[3,4], col3=[4,5])
print(tab.tabulate(tabular_data = data, headers=["", 'Agent1', 'Agent2'], tablefmt='orgtbl'))

data = dict(col1=[1,2], col2=[3,4], col3=[4,5])
print(tab.tabulate(tabular_data = data, headers='keys', tablefmt='rst'))

data = dict(col1=[1,2], col2=[3,4], col3=[4,5])
print(tab.tabulate(tabular_data = data, headers='keys', tablefmt='mediawiki'))

data = dict(col1=[1,2], col2=[3,4], col3=[4,5])
print(tab.tabulate(tabular_data = data, headers='keys', tablefmt='latex'))

data = [['AverageReaward', 100], ['Q_Value', 20]]
print(tab.tabulate( data, headers=["Exp Name", "TD3"], tablefmt="orgtbl"))
