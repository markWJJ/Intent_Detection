import pickle

data_dict={}
for ele in open('../../../corpus_data/意图识别数据_all.txt','r').readlines():
    index=ele.split('\t\t')[0]
    sent=ele.split('\t\t')[1]
    data_dict[index]=sent




dev=pickle.load(open('./dev.p','rb'))
# #
for k,v in dev.items():
    print(k)
    for k_ele,v_ele in v.items():
        if v_ele[0]>=1:
            print(k_ele)
            for ele in v_ele[1]:
                pass
                # print(data_dict[ele[0]])
                # print(ele[2])
            print(k_ele,v_ele)
    print('\n\n')

#
import xlwt
# 创建一个workbook 设置编码
workbook = xlwt.Workbook(encoding = 'utf-8')
# 创建一个worksheet
for k,v in dev.items():
    print(k)
    worksheet = workbook.add_sheet(k)
    # 写入excel
    # 参数对应 行, 列, 值
    index=1
    for k_ele,v_ele in v.items():
        if v_ele[0]>=1:
            worksheet.write(index,1, k_ele)
            for ele in v_ele[1]:
                sent=data_dict[ele[0]]
                ele=str(ele[1])+'_'+ele[2]
                worksheet.write(index,2,ele)
                worksheet.write(index,3,sent)
                index += 1

    # 保存
workbook.save('意图预测结果_4类_20180711_1.xls')

