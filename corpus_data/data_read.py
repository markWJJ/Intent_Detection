# import csv
# #
# # data=csv.reader(open('./保单查询标注意图(1).csv','r').readlines())
# #
# # for ele in data:
# #     print(ele[0],'\t\t',ele[1])
#
# import json
# # for line in open('./待扩展模板同义句_6.12.txt').readlines():
# #     line=line.replace('\n','')
# #     for e in line.split('\t'):
# #         print(e)
#
# fr=(line for line in open('./意图识别数据_all.txt','r').readlines())
# for e in fr:
#     e=e.replace(' ','').replace('\t\t\t\t','\t\t').replace('\n','')
#     print(e)

# import re
# sub_pattern='\\?'
# fw=open('./tt.txt','w')
# for ele in open('sentence.txt','r').readlines():
#     ele=ele.replace('\n','')
#     sent=ele.split('|')[1]
#     label=ele.split('|')[0]
#     fw.write(sent)
#     fw.write('\t\t')
#     fw.write(label)
#     fw.write('\n')


# for ele in open('./tt.txt','r').readlines():
#     for e in ele.split(' '):
#         print(e)
#
# import csv
# dd_dict={}
#
# for ele in open('./FAQ(2).txt','r').readlines():
#     ele=ele.replace('\n','').replace('\t\t','\t')
#     try:
#         sent=ele.split('\t')[0]
#         label=ele.split('\t')[1]
#     except:
#         print(ele)
#     if sent not in dd_dict:
#         dd_dict[sent]=label
#
# for ele in open('./意图识别数据_all.txt', 'r').readlines():
#     ele = ele.replace('\n', '').replace('\t\t', '\t')
#     try:
#         sent = ele.split('\t')[0]
#         label = ele.split('\t')[1]
#     except:
#         print(ele)
#     if sent not in dd_dict:
#         dd_dict[sent] = label
#
# fw=open('./tt.txt','w')
#
# for k,v in dd_dict.items():
#     fw.write(k)
#     fw.write('\t\t')
#     fw.write(v)
#     fw.write('\n')

# import xlrd
#
# work_sheet=xlrd.open_workbook('./dn_data_1.xlsx')
# label_list=['1宏观预测','2产品诊断','3政策规范','4知识方法','5理财规划','6事实分析']
#
# fw=open('./ww.txt','w')
#
# for ele in label_list:
#     sheet=work_sheet.sheet_by_name(ele)
#
#     for i in range(sheet.nrows):
#         fw.write(sheet.cell_value(i,0))
#         fw.write('\t\t')
#         fw.write(ele)
#         fw.write('\n')

# import xlrd
#
# fw=open('FAQ.txt','w')
# work_sheet=xlrd.open_workbook('./意图标注数据_第二批_20180711.xlsx')
#
# sheet=work_sheet.sheet_by_index(0)
#
# index=4963
# for i in range(sheet.nrows):
#     sent=sheet.cell_value(i,0)
#     label=sheet.cell_value(i,2)
#     sent=sent.replace('\n','')
#     print(sent,label)
#     if int(label)==1:
#         fw.write(str(index))
#         fw.write('\t\t')
#         fw.write(sent)
#         fw.write('\t\t')
#         fw.write('1宏观预测')
#         fw.write('\n')
#         index+=1
#     if int(label)==2:
#         fw.write(str(index))
#         fw.write('\t\t')
#         fw.write(sent)
#         fw.write('\t\t')
#         fw.write('2产品诊断')
#         fw.write('\n')
#         index += 1
#     if int(label)==3:
#         fw.write(str(index))
#         fw.write('\t\t')
#         fw.write(sent)
#         fw.write('\t\t')
#         fw.write('3政策规范')
#         fw.write('\n')
#         index += 1
#     if int(label)==4:
#         fw.write(str(index))
#         fw.write('\t\t')
#         fw.write(sent)
#         fw.write('\t\t')
#         fw.write('4知识方法')
#         fw.write('\n')
#         index += 1
#     if int(label)==5:
#         fw.write(str(index))
#         fw.write('\t\t')
#         fw.write(sent)
#         fw.write('\t\t')
#         fw.write('5理财规划')
#         fw.write('\n')
#         index += 1
#     if int(label)==6:
#         fw.write(str(index))
#         fw.write('\t\t')
#         fw.write(sent)
#         fw.write('\t\t')
#         fw.write('6时事分析')
#         fw.write('\n')
#         index += 1


# import xlrd
# fw=open('ww.txt','w')
# work_sheet=xlrd.open_workbook('./20180710.xlsx')
#
# index=0
# for i in range(6):
#     sheet=work_sheet.sheet_by_index(i)
#     for j in range(sheet.nrows):
#         sent=sheet.cell_value(j,0).replace('\n','')
#         if sent:
#             fw.write(str(index))
#             fw.write('\t\t')
#             fw.write(sent)
#             fw.write('\t\t')
#             fw.write(sheet.name)
#             fw.write('\n')
#             index+=1



#
# import xlrd
#
# work_sheet = xlrd.open_workbook('./20180710.xlsx')
#
# sheet=work_sheet.sheet_by_index(0)
#
# index=4705
# for i in range(sheet.nrows):
#     sent=sheet.cell_value(i,0)
#     id=sheet.cell_value(i,2)
#     if sent:
#         if id==1:
#             print(index,'\t\t',sent.replace('\n',''),'\t\t','1宏观预测')
#         elif id==2:
#             print(index,'\t\t',sent.replace('\n',''),'\t\t','2产品诊断')
#         elif id==3:
#             print(index,'\t\t',sent.replace('\n',''),'\t\t','3政策规范')
#         elif id==4:
#             print(index,'\t\t',sent.replace('\n',''),'\t\t','4知识方法')
#         elif id==5:
#             print(index,'\t\t',sent.replace('\n',''),'\t\t','5理财规划')
#         index+=1



# from jieba import analyse
#
# tfidf=analyse.extract_tags
#
# s='黄怎样才算财富自由？我们普通人怎样才做到呢？谢谢'
#
# keys=tfidf(s)
#
# for k in keys:
#     print(k)
# index=8909
# for ele in open('./ww.txt','r').readlines():
#     ele=ele.replace('\n','')
#     print(index,'\t\t',ele,'\t\t','6时事分析')
#
#     index+=1


# import xlrd
# import os
# tab=['1宏观预测','2产品诊断','5理财规划']
#
# work_sheet=xlrd.open_workbook('./20180710.xlsx')
# for ele in tab:
#     fw=open(ele+'.txt','w')
#     sheet=work_sheet.sheet_by_name(ele)
#
#     for i in range(sheet.nrows):
#         sent=sheet.cell_value(i,0).replace('\n','')
#         label=sheet.cell_value(i,1)
#         fw.write(str(i))
#         fw.write('\t\t')
#         fw.write(sent)
#         fw.write('\t\t')
#         fw.write(label)
#         fw.write('\n')


import xlrd

work_sheet=xlrd.open_workbook('./数据标注文档.xlsx')

table=work_sheet.sheet_by_name('Corpus')
index=0
for i in range(table.nrows):

    label=table.cell_value(i,0)
    sent=table.cell_value(i,1)
    if label=='其他':
        label='other'
    print(index,'\t\t',sent,'\t\t',label)
    index+=1