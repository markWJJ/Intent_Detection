#!/usr/bin/python3
# coding: utf-8

import requests
import json
import datetime

ES_HOST = '192.168.3.105'  # 公司内网地址
# ES_HOST = '52.80.177.148'  # 外网30G内存对应的数据库
# ES_HOST = '52.80.187.77'  # 外网，南迪数据管理关联的es数据库；
# ES_HOST = '10.13.70.57'  # 外网词向量测试服
# ES_HOST = '10.13.70.173'  # 外网词向量正式服
# ES_HOST = None
ES_PORT = '9200'
ES_INDEX = 'intent'  # 必须是小写
ES_USER = 'elastic'
ES_PASSWORD = 'webot2008'

intent_entity_dict={'理赔_申请_保障项目' : [['Baoxianchanpin', 'Baozhangxiangmu'], []] ,
'询问_犹豫期' : [['Baoxianchanpin'], []] ,
'询问_保险责任' : [['Baoxianchanpin'], []] ,
'询问_合同_恢复' : [['Baoxianchanpin'], []] ,
'询问_保单_借款' : [['Baoxianchanpin'], []] ,
'询问_特别约定' : [['Baoxianchanpin'], []] ,
'询问_合同_终止' : [['Baoxianchanpin'], []] ,
'询问_合同_成立' : [['Baoxianchanpin'], []] ,
'询问_宣告死亡' : [['Baoxianchanpin'], []] ,
'询问_如实告知' : [['Baoxianchanpin'], []] ,
'询问_合同_解除' : [['Baoxianchanpin'], []] ,
'询问_保险期间' : [['Baoxianchanpin'], []] ,
'询问_保费_缴纳_垫缴' : [['Baoxianchanpin'], []] ,
'询问_受益人' : [['Baoxianchanpin'], []] ,
'询问_体检' : [['Baoxianchanpin'], []] ,
'询问_未归还款项偿还' : [['Baoxianchanpin'], []] ,
'介绍_合同_构成' : [['Baoxianchanpin'], []] ,
'询问_争议处理' : [['Baoxianchanpin'], []] ,
'询问_投保_年龄' : [['Baoxianchanpin'], []] ,
'询问_投保_适用币种' : [['Baoxianchanpin'], []] ,
'询问_保险金额_基本保险金额' : [['Baoxianchanpin'], []] ,
'询问_合同_种类' : [['Baoxianchanpin'], []] ,
'询问_等待期' : [['Baoxianchanpin'], []] ,
'询问_减额缴清' : [['Baoxianchanpin'], []] ,
'询问_保费_缴纳_方式' : [['Baoxianchanpin'], ['Jiaofeifangshi']] ,
'询问_保费_缴纳_年期' : [['Baoxianchanpin'], []] ,
'介绍_服务内容' : [['Fuwuxiangmu'], []] ,
'承保内容_产品_疾病' : [['Baoxianchanpin'], []] ,
'询问_公司_产品' : [['Baozhangxiangmu', 'Baoxianchanpin'], []] ,
'询问_宽限期_时间' : [['Baoxianchanpin'], []] ,
'询问_宽限期_定义' : [['Baoxianchanpin'], []] ,
'询问_产品_优势' : [['Baoxianchanpin'], []] ,
'询问_免赔_额_定义' : [['Baoxianchanpin'], []] ,
'询问_免赔_额_数值' : [['Baoxianchanpin'], []] ,
'询问_保费_缴纳' : [['Baoxianchanpin'], []] ,
'询问_诉讼时效' : [['Baoxianchanpin'], []] ,
'变更_通讯方式' : [['Baoxianchanpin'], []] ,
'询问_保险金_给付' : [['Baoxianchanpin'], []] ,
'区别' : [[], []] ,
'承保_投保_疾病' : [['Baoxianchanpin', 'Jibing'], []] ,
'询问_保单_借款_还款期限' : [['Baoxianchanpin'], []] ,
'承保_投保_情景' : [['Baoxianchanpin', 'Qingjing'], []] ,
'介绍_定义_保险种类' : [[], ['Baoxianzhonglei']] ,
'承保范围_情景' : [['Baoxianchanpin', 'Qingjing'], []] ,
'介绍_定义_疾病种类' : [['Jibingzhonglei', 'Baoxianchanpin'], []] ,
'介绍_定义_保障项目' : [['Baoxianchanpin', 'Baozhangxiangmu'], []] ,
'理赔_申请_疾病种类' : [['Baoxianchanpin', 'Jibingzhonglei'], []] ,
'询问_失踪处理' : [['Baoxianchanpin'], []] ,
'询问_疾病种类_包含_疾病' : [['Baoxianchanpin', 'Jibingzhonglei'], []] ,
'询问_疾病_属于_疾病种类' : [['Baoxianchanpin', 'Jibing', 'Jibingzhonglei'], []] ,
'询问_信息误告' : [['Baoxianchanpin'], []] ,
'询问_疾病种类_包含' : [['Jibingzhonglei'], []] ,
'询问_疾病_预防' : [['Jibing'], []] ,
'询问_疾病_高发疾病' : [['Baoxianchanpin'], []] ,
'询问_体检_异常指标分析' : [[], []] ,
'介绍_定义_体检异常指标' : [[], []] ,
'介绍_身体器官_构成' : [[], []] ,
'询问_疾病_发病原因' : [['Jibing'], []] ,
'询问_体检_体检项目_内容' : [[], []] ,
'询问_投保_途径_使用方法' : [[], []] ,
'询问_合同_领取方式' : [[], ['Baoxianchanpin']] ,
'询问_投保_途径' : [[], ['Baoxianchanpin']] ,
'询问_投保人_被保险人' : [[], []] ,
'区别_疾病种类' : [['Baoxianchanpin'], []] ,
'限制_保障项目_年龄' : [['Baoxianchanpin', 'Baozhangxiangmu'], []] ,
'介绍_公司_经营状况' : [['Baoxianchanpin'], []] ,
'限制_投保_职业' : [['Baoxianchanpin'], []] ,
'询问_产品_优惠' : [[], ['Baoxianchanpin']] ,
'询问_保费_缴纳_方式_保险种类' : [['Baoxianzhonglei'], ['Baoxianchanpin']] ,
'限制_体检时限' : [[], []] ,
'询问_投保_途径_投保流程' : [['Baoxianchanpin'], []] ,
'询问_保费_定价' : [['Baoxianchanpin'], []] ,
'理赔_申请_渠道' : [[], []] ,
'询问_产品_最低保额' : [['Baoxianchanpin'], []] ,
'理赔_缓缴保险费' : [['Baoxianchanpin'], []] ,
'理赔_申请_资料_保障项目' : [[], ['Baozhangxiangmu']] ,
'介绍_公司_股东构成' : [[], []] ,
'理赔_重复理赔_保障项目' : [['Baozhangxiangmu', 'Baoxianchanpin'], []] ,
'询问_产品_价格' : [['Baoxianchanpin'], []] ,
'介绍_公司' : [[], []] ,
'询问_产品_属于_保险种类' : [['Baoxianchanpin', 'Baoxianzhonglei'], []] ,
'承保_区域' : [['Baoxianchanpin'], []] ,
'询问_保障项目_赔付_次数' : [['Baoxianchanpin', 'Baozhangxiangmu'], []] ,
'询问_保费_缴纳_渠道' : [['Baoxianchanpin'], []] ,
'询问_财务问卷-对象' : [[], []] ,
'询问_疾病种类_治疗费用' : [['Jibingzhonglei'], []] ,
'限制_投保_特殊人群' : [['Baoxianchanpin'], []] ,
'询问_保险金额_推荐金额' : [['Baoxianchanpin'], []] ,
'询问_核保_未通过_解决办法' : [[], []] ,
'询问_体检_标准' : [[], []] ,
'询问_投保_资料' : [['Baoxianchanpin'], []] ,
'询问_保险条款' : [['Baoxianchanpin'], []] ,
'理赔_医院' : [['Baoxianchanpin'], ['Yiyuan']] ,
'询问_保障项目_赔付' : [['Baoxianchanpin', 'Baozhangxiangmu'], []] ,
'介绍_产品_推荐' : [['Baoxianchanpin'], []] ,
'限制_工具使用' : [[], []] ,
'询问_保险类型' : [['Baoxianchanpin'], []] ,
'询问_诊断报告_领取' : [[], []] ,
'介绍_产品_升级' : [['Baoxianchanpin'], []] ,
'询问_体检_指引' : [['Baoxianchanpin'], []] ,
'询问_疾病种类_生存期' : [['Jibingzhonglei', 'Baoxianchanpin'], []] ,
'限制_地域_理赔' : [[], []] ,
'限制_投保_特殊人群_保额' : [['Baoxianchanpin'], []] ,
'询问_权益关系' : [['Baoxianchanpin'], []] ,
'承保_范围_保障项目' : [['Baoxianchanpin', 'Baozhangxiangmu'], []] ,
'介绍_产品' : [[], []] ,
'询问_合同_纸质合同_申请' : [[], []] ,
'询问_疾病种类_赔付流程' : [['Baoxianchanpin', 'Baoxianzhonglei'], []] ,
'理赔_速度' : [['Baoxianchanpin'], []] ,
'限制_地域_产品保障' : [['Baoxianchanpin'], []] ,
'询问_合同_丢失' : [[], []] ,
'介绍_合同_电子合同' : [[], []] ,
'询问_保单_查询' : [[], []] ,
'理赔_条件' : [['Baoxianchanpin', 'Baozhangxiangmu'], []] ,
'理赔_方式' : [['Baoxianchanpin', 'Baozhangxiangmu'], []] ,
'询问_投保_指引' : [['Baoxianchanpin'], []] ,
'询问_业务_材料' : [[], []] ,
'限制_投保人' : [[], []] ,
'限制_红利申请' : [[], []] ,
'询问_合同_终止_情景' : [[], ['Qingjing']] ,
'理赔_报案_目的' : [[], []] ,
'询问_险种关联_申请资料' : [[], []] ,
'变更_投保人' : [['Baoxianchanpin'], []] ,
'询问_查询方式' : [[], []] ,
'理赔_申请_疾病' : [['Baoxianchanpin', 'Jibing'], []] ,
'赔付_疾病_保障项目' : [['Baoxianchanpin', 'Jibing', 'Baozhangxiangmu'], []] ,
'赔付_情景_保障项目' : [['Baoxianchanpin', 'Qingjing', 'Baozhangxiangmu'], []] ,
'介绍_定义_情景' : [['Qingjing'], []] ,
'介绍_定义_疾病' : [['Jibing', 'Baoxianchanpin'], []] ,
'询问_保险事故通知' : [['Baoxianchanpin'], []] ,
'询问_疾病_包含_疾病' : [['Jibing'], []] ,
'介绍_定义_释义项' : [['Shiyi'], []] ,
# '询问_公司_分公司_联系方式' : [[], []] ,
'询问_复效期' : [['Baoxianchanpin'], []] ,
'理赔_比例' : [['Baoxianchanpin', 'Baozhangxiangmu'], []] ,
'限制_投保' : [['Baoxianchanpin'], []] ,
'询问_投保人' : [['Baoxianchanpin'], []] ,
'变更_缴费方式' : [['Baoxianchanpin'], []] ,
'询问_现金价值' : [['Baoxianchanpin'], []] ,
'询问_保险种类' : [['Baoxianchanpin'], ['Baoxianzhonglei']] ,
'介绍_定义_产品' : [[], ['Baoxianzhonglei']] ,
'承保_范围_疾病' : [['Baoxianchanpin', 'Jibing'], []] ,
'询问_复效期_滞纳金' : [['Baoxianchanpin'], []] ,
'询问_减额缴清_影响' : [['Baoxianchanpin'], []] ,
'限制_保险种类_年龄' : [['Baoxianzhonglei'], []] ,
'询问_保费_缴纳_忘缴' : [[], []] ,
'限制_购买_产品' : [['Baoxianchanpin'], []] ,
'理赔_申请_特殊情况' : [[], ['Baoxianchanpin']] ,
'询问_现金价值_豁免保费' : [['Baoxianchanpin'], []] ,
'询问_投保_途径_月缴首期' : [[], []] ,
'询问_保障项目_赔付_赔付一半' : [['Baozhangxiangmu'], []] ,
'理赔_方式_特殊人群' : [['Baoxianchanpin'], []] ,
'限制_国外医院' : [[], []] ,
'介绍_投资连结保险投资账户' : [[], []] ,
'询问_增值服务_项目' : [['Baoxianchanpin', 'Fuwuxiangmu'], []] ,
'询问_增值服务_内容' : [['Baoxianchanpin', 'Fuwuxiangmu'], []] ,
'询问_合作单位_合作医院' : [[], []] ,
'询问_保险金额' : [['Baoxianchanpin'], []] ,
'介绍_定义_公司' : [[], []] ,
'询问_免赔_额_数值_保险种类' : [['Baoxianzhonglei'], []] ,
'询问_合同_附加险' : [['Baoxianchanpin'], []] ,
'询问_保险金额_保额累计' : [['Baoxianchanpin'], []] ,
'询问_投保书_填写' : [[], []] ,
'承保_范围_产品_疾病' : [['Baoxianchanpin'], []] ,
'变更_合同' : [['Baoxianchanpin'], []] ,
'询问_引导' : [[], []] ,
'询问_保费_支出' : [[], []] ,
'询问_公司_地址' : [[], []] ,
'询问_联系方式' : [['Didian'], []] ,
'询问_合作单位_合作银行' : [[], []] ,
'询问_商业保险与医保的关系' : [['Baoxianchanpin'], []] ,
'询问_保单_回溯' : [[], ['保单回溯']] ,
'询问_体检_意义' : [[], []] ,
'询问_合作单位_中信银行' : [[], []] ,
'询问_产品_对比' : [['Baoxianchanpin'], []] ,
'询问_疾病_治疗费用' : [[], []] ,
'询问_增值服务_使用次数' : [[], []] ,
'询问_增值服务_使用时间' : [['Baoxianchanpin'], []] ,
'询问_增值服务_亮点' : [[], []] ,
'变更通讯资料相关规定' : [[], []] ,
'银行转账授权相关规定' : [[], []] ,
'变更投保人相关规定' : [[], []] ,
'更改个人身份资料相关规定' : [[], []] ,
'变更签名相关规定' : [[], []] ,
'保费逾期未付选择相关规定' : [[], []] ,
'变更受益人相关规定' : [[], []] ,
'领取现金红利相关规定' : [[], []] ,
'身故保险金分期领取选择相关规定' : [[], []] ,
'变更红利领取方式相关规定' : [[], []] ,
'指定第二投保人相关规定' : [[], []] ,
'复效相关规定' : [[], []] ,
'变更职业等级相关规定' : [[], []] ,
'变更缴费方式相关规定' : [[], []] ,
'结束保险费缓缴期相关规定' : [[], []] ,
'降低主险保额相关规定' : [[], []] ,
'变更附加险相关规定' : [[], []] ,
'减额缴清相关规定' : [[], []] ,
'补充告知相关规定' : [[], []] ,
'取消承保条件相关规定' : [[], []] ,
'保单借款相关规定' : [[], []] ,
'保单还款相关规定' : [[], []] ,
'生存给付确认相关规定' : [[], []] ,
'变更年金领取方式相关规定' : [[], []] ,
'变更生存保险金领取方式相关规定' : [[], []] ,
'领取生存保险金相关规定' : [[], []] ,
'变更给付账号相关规定' : [[], []] ,
'满期给付生存确认相关规定' : [[], []] ,
'险种关联选择相关规定' : [[], []] ,
'部分提取相关规定' : [[], []] ,
'额外投资相关规定' : [[], []] ,
'投资账户选择相关规定' : [[], []] ,
'投资账户转换相关规定' : [[], []] ,
'终止保险合同相关规定' : [[], []] ,
'申请定期额外投资相关规定' : [[], []] ,
'变更定期额外投资相关规定' : [[], []] ,
'终止定期额外投资相关规定' : [[], []] ,
'犹豫期减保相关规定' : [[], []] ,
'犹豫期终止合同相关规定' : [[], []] ,
'犹豫期其他保全变更相关规定' : [[], []] ,
'补发保单相关规定' : [[], []] ,
'变更通讯资料申请时间' : [[], []] ,
'银行转账授权申请时间' : [[], []] ,
'变更投保人申请时间' : [[], []] ,
'更改个人身份资料申请时间' : [[], []] ,
'变更签名申请时间' : [[], []] ,
'保费逾期未付选择申请时间' : [[], []] ,
'变更受益人申请时间' : [[], []] ,
'领取现金红利申请时间' : [[], []] ,
'身故保险金分期领取选择申请时间' : [[], []] ,
'变更红利领取方式申请时间' : [[], []] ,
'指定第二投保人申请时间' : [[], []] ,
'复效申请时间' : [[], []] ,
'变更职业等级申请时间' : [[], []] ,
'变更缴费方式申请时间' : [[], []] ,
'结束保险费缓缴期申请时间' : [[], []] ,
'降低主险保额申请时间' : [[], []] ,
'变更附加险申请时间' : [[], []] ,
'减额缴清申请时间' : [[], []] ,
'补充告知申请时间' : [[], []] ,
'取消承保条件申请时间' : [[], []] ,
'保单借款申请时间' : [[], []] ,
'保单还款申请时间' : [[], []] ,
'生存给付确认申请时间' : [[], []] ,
'变更年金领取方式申请时间' : [[], []] ,
'变更生存保险金领取方式申请时间' : [[], []] ,
'领取生存保险金申请时间' : [[], []] ,
'变更给付账号申请时间' : [[], []] ,
'满期给付生存确认申请时间' : [[], []] ,
'险种关联选择申请时间' : [[], []] ,
'部分提取申请时间' : [[], []] ,
'额外投资申请时间' : [[], []] ,
'投资账户选择申请时间' : [[], []] ,
'投资账户转换申请时间' : [[], []] ,
'终止保险合同申请时间' : [[], []] ,
'申请定期额外投资申请时间' : [[], []] ,
'变更定期额外投资申请时间' : [[], []] ,
'终止定期额外投资申请时间' : [[], []] ,
'犹豫期减保申请时间' : [[], []] ,
'犹豫期终止合同申请时间' : [[], []] ,
'犹豫期其他保全变更申请时间' : [[], []] ,
'补发保单申请时间' : [[], []] ,
'保险产品支持保全项' : [['Baoxianchanpin'], []] ,
'保险产品变更通讯资料的范围' : [['Baoxianchanpin'], []] ,
'保险产品银行转账授权的范围' : [['Baoxianchanpin'], []] ,
'保险产品更改个人身份资料的范围' : [['Baoxianchanpin'], []] ,
'保险产品变更签名的范围' : [['Baoxianchanpin'], []] ,
'保险产品变更受益人的范围' : [['Baoxianchanpin'], []] ,
'保险产品变更投保人的范围' : [['Baoxianchanpin'], []] ,
'保险产品保费逾期未付选择的范围' : [['Baoxianchanpin'], []] ,
'保险产品补发保单的范围' : [['Baoxianchanpin'], []] ,
'保险产品变更职业等级的范围' : [['Baoxianchanpin'], []] ,
'保险产品变更缴费方式的范围' : [['Baoxianchanpin'], []] ,
'保险产品变更保险计划的范围' : [['Baoxianchanpin'], []] ,
'保险产品复效的范围' : [['Baoxianchanpin'], []] ,
'保险产品减额缴清的范围' : [['Baoxianchanpin'], []] ,
'保险产品取消承保条件的范围' : [['Baoxianchanpin'], []] ,
'保险产品补充告知的范围' : [['Baoxianchanpin'], []] ,
'保险产品保单借款的范围' : [['Baoxianchanpin'], []] ,
'保险产品保单还款的范围' : [['Baoxianchanpin'], []] ,
'保险产品终止保险合同的范围' : [['Baoxianchanpin'], []] ,
'保险产品附约变更的范围' : [['Baoxianchanpin'], []] ,
'保险产品满期生存确认的范围' : [['Baoxianchanpin'], []] ,
'保险产品降低主险保额的范围' : [['Baoxianchanpin'], []] ,
'询问_产品_续保' : [['Baoxianchanpin'], []] ,
'询问_产品_期满返还' : [['Baoxianchanpin'], []] ,
'询问_产品_费用与报销' : [['Baoxianchanpin'], []] ,
'变更_保险金额' : [['Baoxianchanpin'], []] ,
'限制_保险金额' : [['Baoxianchanpin'], []] ,
'理赔_情景_全残或身故' : [['Qingjing', 'Baoxianchanpin'], []] ,
'承保_范围_保障项目_疾病' : [['Baozhangxiangmu'], []] ,
'询问_责任免除_条款' : [['Baoxianchanpin'], []] ,
'询问_责任免除_保障项目' : [['Baoxianchanpin'], []] ,
'询问_疾病_属于_产品' : [['Baoxianchanpin'], []] ,
# '承保_范围_产品_疾病' : [['Baoxianchanpin'], []] ,
'承保_范围_情景' : [['Baoxianchanpin', 'Jibing'], []] ,
'询问_豁免_可豁免' : [['Baoxianchanpin', 'Jibing'], []] ,
'理赔_申请_资料' : [['Baoxianchanpin'], []] ,
'询问_保费_核算' : [['Baoxianchanpin'], []] ,
'变更_保全' : [[], []] ,
'询问_豁免_不可豁免' : [['Baoxianchanpin', 'Jibing'], []] ,
'理赔_重复理赔' : [['Baoxianchanpin', 'Jibing'], []] ,
'询问_补偿原则':[[],[]],
'询问_产品_续保_年限':[['Baoxianchanpin'],[]],
'询问_ 无保险事故_优惠':[[],[]],
'询问_保费_费率_调整':[[],[]],
'询问_附加险_规则':[[],[]],
'询问_合同_解除_时限':[[],[]],
'询问_公司_总部':[[],[]],
'询问_公司_区域范围':[['Didian'],[]],
'询问_服务时间':[['Didian'],[]],
'询问_保全规则':[[],[]],
'询问_保单_代理人信息':[[],[]],
'介绍_公司_名称':[[],[]],
'介绍_产品_名称':[[],[]],
'区别_保险责任':[[],[]],
'区别_投保年龄':[[],[]],
'区别_责任免除':[[],[]],
'区别_现金价值':[[],[]],
'区别_重疾种类':[[],[]],
'区别_重疾定义':[[],[]],
'区别_缴费期限':[[],[]],
'区别_保险金额_限制':[[],[]],
'区别_保额累计_限制':[[],[]],
'询问_地址':[['Didian'],[]],
}

for k in intent_entity_dict.keys():
    print(k,'\t',':','\t')
def download_template_intent(es_host=ES_HOST, es_port=ES_PORT, _index='templates_question', es_user=ES_USER, es_password=ES_PASSWORD, pid='all_baoxian'):
    """
    模板里头有必须实体，可选实体，以模板数据为准
    :param es_host: 
    :param es_port: 
    :param _index: 
    :param _type:

    :param es_user: 
    :param es_password: 
    :param data: 
    :param pid: 
    :return: 
    """
    es_index_alias = "{}_{}_alias".format(pid.lower(), _index)
    intent_bixuan_kexuan_dict = {}
    intent_list = list(intent_entity_dict.keys())
    try:
        # 获取全部的模板数据
        url = 'http://{}:{}/{}/_search?scroll=10m&size=5000'.format(es_host, es_port, es_index_alias)

        args_json = {
            "query" : {
                "match_all" : {}
            }
        }
        r = requests.get(url, json=args_json, auth=(es_user, es_password))
        ret = r.json()
        hits = ret['hits']['hits']
        datas = [h['_source'] for h in hits]

    except Exception as e:
        print("在索引`{}:{}/{}`下获取意图为`{}`的必须、可选参数出错： \n{}".format(ES_HOST, ES_PORT, es_index_alias, intent_list, e))
        datas = []

    for data in datas:
        intent = data.get('intent')
        pass_intent_list = []
        assert intent in intent_list or intent in pass_intent_list, "模板中的意图`{}`应该在意图字典中存在".format(intent)
        bixuan = data.get('必选实体', [])
        kexuan = data.get('可选实体', [])
        if intent and (bixuan or kexuan):
            intent_bixuan_kexuan_dict.setdefault(intent, (bixuan, kexuan))

    return intent_bixuan_kexuan_dict


def intent_to_es(es_host=ES_HOST, es_port=ES_PORT, _index=ES_INDEX, _type='intent', es_user=ES_USER, es_password=ES_PASSWORD, data=None, pid='all_baoxian'):

    # 模板中存在的意图集合
    templates_intent_bixuan_kexuan_dict = download_template_intent(es_host=es_host, es_port=es_port,
                                                                   _index='templates_question', es_user=es_user,
                                                                   es_password=es_password, pid=pid)

    es_index = "{}_{}".format(pid, _index)
    now = datetime.datetime.now()
    index_end = now.strftime('%Y%m%d_%H%M%S')
    current_es_index = "{}_{}".format(es_index, index_end).lower()

    alias_name = '{}_alias'.format(es_index)

    url = "http://{}:{}/{}/{}/_bulk".format(es_host, es_port, current_es_index, _type)

    all_data = ''
    # del_alias_name = {"delete": {"_index": alias_name}}
    # all_data += json.dumps(del_alias_name, ensure_ascii=False) + '\n'
    for template_id, (intent, entity_list) in enumerate(data.items()):
        # 若意图在模板库中存在，则以模板的意图为准
        if templates_intent_bixuan_kexuan_dict.get(intent):
            bixuan, kexuan = templates_intent_bixuan_kexuan_dict.get(intent)
        else:
            bixuan, kexuan = entity_list
        doc = {"intent": intent, "必选实体": bixuan, "可选实体": kexuan, '模板id': template_id}
        create_data = {"create": {"_id": template_id}}
        all_data += json.dumps(create_data, ensure_ascii=False) + '\n'
        all_data += json.dumps(doc, ensure_ascii=False) + '\n'

    ret = requests.post(url=url, data=all_data.encode('utf8'), auth=(es_user, es_password))
    # print(ret.json())

    # 添加别名
    data = {
        "actions": [
            {"remove": {
                "alias": alias_name,
                "index": "_all"
            }},
            {"add": {
                "alias": alias_name,
                "index": current_es_index
            }}
        ]
    }
    url = "http://{}:{}/_aliases".format(es_host, es_port)
    r = requests.post(url, json=data, auth=(es_user, es_password))
    print(r.json())

def create_one(_id, intent, bixuan, kexuan, es_host=ES_HOST, es_port=ES_PORT, _index=ES_INDEX, _type='intent', pid='all_baoxian'):
    """
    向意图表中插入单条数据
    :param _id: 
    :param intent: 
    :param bixuan: 
    :param kexuan: 
    :return: 
    """
    es_index = "{}_{}".format(pid, _index)
    alias_name = '{}_alias'.format(es_index)

    url = "http://{}:{}/{}/{}/{}".format(es_host, es_port, alias_name, _type, _id)
    template_id = _id
    doc = {"intent": intent, "必选实体": bixuan, "可选实体": kexuan, '模板id': template_id}
    r = requests.post(url, json=doc, auth=(ES_USER, ES_PASSWORD))
    print(r.json())

def main():
    # es_user = 'elastic'
    # es_password = 'webot2008'
    # es_host = '192.168.3.145'
    # es_port = '9200'
    # _index = 'intent'
    # _type = 'intent'

    intent_to_es(es_host=ES_HOST, es_port=ES_PORT, _index=ES_INDEX, _type='intent', es_user=ES_USER,
                 es_password=ES_PASSWORD, data=intent_entity_dict, pid='zhongdeanlian')


    # 插入单条：
    # _id = 144
    # intent = '测试意图'
    # bixuan = ['Jibing']
    # kexuan = []
    # create_one(_id, intent, bixuan, kexuan, _index=ES_INDEX, _type='intent', pid='all_baoxian')

if __name__ == '__main__':
    main()
