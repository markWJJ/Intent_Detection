

import xlrd
import os
tab=['1宏观预测','2产品诊断','3政策规范','5理财规划','4知识方法','6时事分析']
index=0
work_sheet=xlrd.open_workbook('./意图标注_20180719.xlsx')
fw = open('ww.txt', 'w')

for ele in tab:
    sheet=work_sheet.sheet_by_name(ele)

    for i in range(sheet.nrows):
        sent=sheet.cell_value(i,0).replace('\n','')
        label_level_1=ele
        try:
            label_level_2=sheet.cell_value(i,1)
        except Exception as ex:
            label_level_2='None'
        if not label_level_2:
            label_level_2='None'
        try:
            label_level_3=sheet.cell_value(i,2)
            if not label_level_3:
                label_level_3='None'
        except Exception:
            label_level_3='None'

        fw.write(str(index))
        fw.write('\t\t')
        fw.write(sent)
        fw.write('\t\t')
        fw.write(label_level_1)
        fw.write('##')
        fw.write(label_level_2)
        fw.write('##')
        fw.write(label_level_3)
        fw.write('\n')
        index+=1