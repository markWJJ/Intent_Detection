#!/usr/bin/python3
# coding: utf-8
import json

import requests

import requests
import time
url = "http://192.168.3.190:8083/nlp/intent"


datas = ['万科技术面分析']

start_time = time.time()

# datas=['我想买保险']
intent_ret = requests.post(url, json={'data': datas})
res = intent_ret.json()

print(res)

end_time=time.time()

print(end_time-start_time)



# import xlrd
#
# worksheet=xlrd.open_workbook('./FAQ测试集.xlsx')
#
# table=worksheet.sheet_by_index(0)
#
# data=[]
# for i in range(table.nrows):
#     s=table.cell_value(i,0)
#     data.append(s)
#
# intent_ret = requests.post(url, json={'data': data})
# res = intent_ret.json()
#
# for k,v in res.items():
#     print(k,'\t\t',v[0][0])
