

# fw=open('dev_en.txt','w')
# for e in open('./dev_new.txt','r').readlines():
#     e=e.replace('\n','')
#     sents=e.split(',')
#     fw.write(sents[0])
#     fw.write('\t\t')
#     fw.write(sents[1])
#     fw.write('\t\t')
#     fw.write(sents[2])
#     fw.write('\n')

s=[1,2,3,4,5]
from functools import reduce

f=reduce(lambda x,y:max(x,y),s)

alist = [[1,2],3,4]
blist = alist
blist[0]=1
print(alist)

# names = ['Amir', 'Betty', 'Chales', 'Tao']
# names.index("Edward")

def addItem(listParam):
    listParam += [1]
mylist = [1, 2, 3, 4]
addItem(mylist)
print(mylist)

list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
print(list1+list2

      )


x=True;y,z=False,False

if x or y and z:
 print('yes')
else:
 print('no')


def myfoo(x, y, z, a):
 return x + z

nums = [1, 2, 3, 4]
print(myfoo(*nums))