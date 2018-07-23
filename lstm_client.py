from xmlrpc.client import  ServerProxy
from sklearn.metrics import classification_report,precision_recall_fscore_support
import time

svr_pipeline=ServerProxy("http://192.168.3.132:8086")

start_time = time.time()

s=['你们的总部']
res = svr_pipeline.intent(s)
print(res)
end_time=time.time()

print(end_time-start_time)


# ss=[]
# for ele in open('./信诚客户问题7-18日.txt').readlines():
#     ele=ele.replace('\n','')
#     ss.append(ele)
#
# result=svr_pipeline.intent(ss)
#
#
# fw=open('./信诚客户问题7-18日_out.txt','w')
# for k,v in result.items():
#     fw.write(k)
#     fw.write('\t\t')
#     fw.write(str(v[0][0]))
#     fw.write('\n')
#     # print(k,v[0][0])
