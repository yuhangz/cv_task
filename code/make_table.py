import json5
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--log')
args = parser.parse_args()

log_path = args.log

for line in open(log_path, 'r', encoding='utf-8'):
    if 'result' in line:
        spl = line.split('result')
        name = spl[0].split()[-1].strip(',')
        result = spl[-1].strip()
        data: dict = eval(result)
        best_param = sorted(data.keys(), key=lambda k : data[k][0] * 1000 - data[k][1], reverse=True)[0]
        for param_name in data:
            performance = data[param_name]
            yn = 'y' if param_name == best_param else 'n'
            print(f'| {name} | {param_name} | {performance} | {yn} |')