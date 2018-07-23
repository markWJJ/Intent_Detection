import pickle





print('#'*100)
dev=pickle.load(open('./train.p','rb'))

for k,v in dev.items():
    print(k)
    for k_ele,v_ele in v.items():
        if v_ele[0]>=1:
            print(k_ele,v_ele)

    print('\n')