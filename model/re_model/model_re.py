import re
import logging
import jieba
import os
import time
import xlrd
import gc
import os,sys
PATH=os.path.split(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0])[0]
sys.path.append(PATH)
type_list=[e.replace('\n','') for e in open(PATH+'/entity_type.txt').readlines() if e]
entity_dict={}
for ele in type_list:
    jieba.load_userdict( PATH+'/entity_data/%s.txt'%ele)
    ele_list=[e.replace('\n','') for e in open(PATH+'/entity_data/%s.txt'%ele,'r')]
    entity_dict[ele]=ele_list
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger=logging.getLogger("data")


jb=[e.replace('\n','').replace("'","").replace('(','').replace(')','') for e in open(PATH+'/entity_data/Jibing.txt','r')] #疾病
bzxm=[e.replace('\n','') for e in open(PATH+'/entity_data/Baozhangxiangmu.txt','r')] #保障项目
bxzl=[e.replace('\n','') for e in open(PATH+'/entity_data/Baoxianzhonglei.txt','r')] #保险种类
bxcp=[e.replace('\n','') for e in open(PATH+'/entity_data/Baoxianchanpin.txt','r')] #保险产品
jbzl=[e.replace('\n','') for e in open(PATH+'/entity_data/Jibingzhonglei.txt','r')] #疾病种类
qj=[e.replace('\n','') for e in open(PATH+'/entity_data/Qingjing.txt','r')] #情景
dd=[e.replace('\n','') for e in open(PATH+'/entity_data/Didian.txt','r')] #地点
yy=[e.replace('\n','') for e in open(PATH+'/entity_data/Yiyuan.txt','r')] #医院
jffs=[e.replace('\n','') for e in open(PATH+'/entity_data/Jiaofeifangshi.txt','r')] #缴费方式
yydj=[e.replace('\n','') for e in open(PATH+'/entity_data/Yiyuandengji.txt','r')] #医院等级
bxj=[e.replace('\n','') for e in open(PATH+'/entity_data/Baoxianjin.txt','r')] #保险金
sy=[e.replace('\n','') for e in open(PATH+'/entity_data/Shiyi.txt','r')] #释义
bqx=[e.replace('\n','') for e in open(PATH+'/entity_data/Baoquanxiang.txt','r')] #保全项




s=[1,2,3]
s.extend()



label_dict={'D1': '承保_范围_疾病', 'D3': '询问_责任免除_保障项目', 'D4': '变更_合同', 'D5': '理赔_申请_保障项目', 'D6': '询问_犹豫期', 'D7': '询问_保险责任',
            'D8': '询问_合同_恢复', 'D10': '询问_保单_借款', 'D11': '询问_特别约定', 'D12': '询问_合同_终止', 'D13': '询问_合同_成立',
            'D14': '询问_宣告死亡', 'D15': '询问_如实告知', 'D16': '询问_合同_解除', 'D17': '询问_保险期间', 'D18': '询问_保费_缴纳_垫缴',
            'D20': '询问_受益人', 'D21': '询问_体检', 'D22': '询问_未归还款项偿还', 'D23': '介绍_合同_构成', 'D24': '询问_争议处理',
            'D25': '询问_投保_年龄', 'D26': '询问_投保_适用币种', 'D27': '询问_保险金额_基本保险金额', 'D28': '询问_合同_种类',
            'D29': '询问_等待期', 'D30': '询问_减额缴清', 'D31': '询问_保费_缴纳_方式', 'D32': '询问_保费_缴纳_年期',
            'D34': '承保内容_产品_疾病', 'D35': '询问_公司_产品', 'D36': '询问_宽限期_定义', 'D37': '询问_产品_优势',
            'D39': '询问_免赔_额_定义', 'D40': '询问_保费_缴纳', 'D42': '询问_诉讼时效', 'D43': '变更_通讯方式',
            'D44': '询问_保险金_给付', 'D47': 'NONe', 'D48': '承保_投保_疾病', 'D49': '询问_保单_借款_还款期限', 'D50': '承保_投保_情景',
            'D51': '介绍_定义_保险种类', 'D52': '承保范围_情景', 'D53': '介绍_定义_疾病种类', 'D54': '介绍_定义_保障项目', 'D55': '理赔_申请_疾病种类',
            'D56': '询问_失踪处理', 'D57': '询问_疾病种类_包含_疾病', 'D58': '询问_疾病_属于_疾病种类', 'D59': '询问_信息误告',
            'D60': '询问_疾病种类_包含', 'D61': '询问_疾病_预防', 'D62': '询问_疾病_高发疾病', 'D63': '询问_体检_异常指标分析',
            'D64': '介绍_定义_体检异常指标', 'D65': '介绍_身体器官_构成', 'D66': '询问_疾病_发病原因', 'D67': '询问_体检_体检项目_内容',
            'D68': '询问_投保_途径_使用方法', 'D69': '询问_合同_领取方式', 'D70': '询问_投保_途径', 'D71': '询问_投保人_被保险人',
            'D72': '区别_疾病种类', 'D73': '限制_保障项目_年龄', 'D74': '介绍_公司_经营状况', 'D75': '限制_投保_职业',
            'D76': '询问_产品_优惠', 'D77': '询问_保费_缴纳_方式_保险种类', 'D78': '限制_体检时限', 'D79': '询问_投保_途径_投保流程',
            'D80': '询问_保费_核算', 'D81': '理赔_申请_渠道', 'D82': '询问_产品_最低保额', 'D83': 'NONe', 'D84': '理赔_缓缴保险费',
            'D85': '理赔_申请_资料_保障项目', 'D86': '介绍_公司_股东构成', 'D87': '理赔_重复理赔_保障项目', 'D88': '询问_产品_价格',
            'D89': '介绍_公司', 'D90': '询问_产品_属于_保险种类', 'D91': '承保_区域', 'D92': '询问_保障项目_赔付_次数', 'D93':'询问_保费_缴纳_渠道', 'D94': '询问_财务问卷-对象', 'D95': '询问_疾病种类_治疗费用', 'D96': '限制_投保_特殊人群',
            'D97': '询问_保险金额_推荐金额', 'D98': '询问_核保_未通过_解决办法', 'D99': '询问_体检_标准', 'D100': '询问_投保_资料',
            'D101': '询问_保险条款', 'D102': '理赔_医院', 'D103': '询问_保障项目_赔付', 'D104': '介绍_定义_保险种类', 'D105': '介绍_产品_推荐',
            'D106': '限制_工具使用', 'D107': '询问_保险类型', 'D108': '询问_诊断报告_领取', 'D109': '询问_产品_优惠', 'D110': '介绍_产品_升级',
            'D111': '询问_体检_指引', 'D112': '询问_保险金额', 'D113': 'NONE', 'D114': '询问_疾病种类_生存期', 'D115': '限制_地域_理赔',
            'D116': '询问_保单_借款_还款期限', 'D117': '限制_投保_特殊人群_保额', 'D118': '询问_权益关系', 'D119': '承保_范围_保障项目',
            'D120': '介绍_产品', 'D121': '询问_合同_纸质合同_申请', 'D122': '询问_疾病种类_赔付流程', 'D123': '理赔_速度', 'D124': '限制_地域_产品保障',
            'D125': '询问_合同_丢失', 'D126': '介绍_合同_电子合同', 'D127': '询问_保单_查询', 'D128': '理赔_条件', 'D129': '理赔_方式',
            'D130': '询问_投保_指引', 'D131': '询问_业务_材料', 'D132': '询问_投保人_被保险人', 'D133': '限制_投保人', 'D134': '限制_红利申请',
            'D135': '询问_合同_终止_情景', 'D136': '理赔_报案_目的', 'D137': '询问_险种关联_申请资料', 'D138': '变更_投保人', 'D139': '询问_查询方式',
            'D140': '理赔_申请_疾病', 'D141': '赔付_疾病_保障项目', 'D142': '赔付_情景_保障项目', 'D143': '介绍_定义_情景', 'D144': '介绍_定义_疾病',
            'D145': '询问_保险事故通知', 'D146': '询问_疾病_包含_疾病', 'D147': 'NONE', 'D148': '介绍_定义_释义项', 'D149': '询问_公司_分公司_联系方式',
            'D150': '询问_复效期', 'D151': '理赔_比例', 'D152': '限制_投保', 'D153': '询问_保险金额_保额累计', 'D154': '询问_投保人',
            'D155': '变更_缴费方式', 'D156': '询问_合同_附加险', 'D157': 'NONE', 'D158': '询问_现金价值', 'D160': '询问_投保书_填写',
            'D161': '询问_增值服务_内容', 'D162': '询问_保险种类', 'D163': '询问_免赔_额_数值', 'D164': '承保_范围_疾病', 'D165': '询问_复效期_滞纳金',
            'D166': '询问_减额缴清_影响', 'D167': '询问_保险金额', 'D168': '限制_保险种类_年龄', 'D169': '承保_范围_疾病', 'D170': '询问_疾病_高发疾病',
            'D171': '询问_保险期间', 'D172': '询问_保费_缴纳_忘缴', 'D173': '限制_购买_产品', 'D174': '理赔_申请_特殊情况', 'D175': '询问_现金价值_豁免保费',
            'D176': '询问_投保_途径_月缴首期', 'D177': '询问_保障项目_赔付_赔付一半', 'D178': '理赔_方式_特殊人群', 'D179': '限制_国外医院',
            'D180': '理赔_申请_保障项目', 'D181': '介绍_投资连结保险投资账户', 'D182': '介绍_定义_产品', 'D183': '询问_增值服务_项目',
            'D184': '介绍_产品', 'D185': '询问_合作单位_合作医院', 'D186': '询问_产品_最低保额', 'D187': '询问_保险金额', 'D188': '询问_宽限期_时间',
            'D189': '介绍_定义_公司', 'D190': '询问_免赔_额_数值_保险种类', 'D192': '询问_引导', 'D193': '询问_保费_支出', 'D194':'询问_公司_地址', 'D195': '询问_联系方式', 'D196': '询问_合作单位_合作银行', 'D197': '询问_商业保险与医保的关系',
            'D199': '询问_体检_意义', 'D200': '询问_合作单位_中信银行', 'D33': '介绍_服务内容', 'D201': '介绍_产品', 'D203': '询问_疾病_治疗费用',
            'D204': '询问_保费_核算', 'D205': '询问_增值服务_使用次数', 'D206': '询问_增值服务_使用时间', 'D207': '询问_增值服务_亮点',
            'D208': '询问_保险责任', 'D209': '询问_产品_续保', 'D210': '询问_产品_期满返还', 'D211': '询问_产品_费用与报销',
            'D212': '询问_诉讼时效', 'D213': '变更_保险金额', 'D214': '理赔_情景_全残或身故', 'D215': '承保_内容_保障项目_疾病',
            'D216': '询问_责任免除_条款', 'D217': '询问_豁免_不可豁免', 'D218': '理赔_申请_资料', 'D219': '询问_豁免_可豁免', 'D222': '理赔_重复理赔',
            'D223':'正则测试意图','D224':'询问_补偿原则','D225':'询问_公司_区域范围','D226':'询问_保单_代理人信息','D227':'询问_保单_缴纳'
            }

del_id={'D2','D38','D19','D38','D41','D45','D46'}

#冲突列表
conflit_list=[
    [('询问体检或验尸','某种体检异常指标分析'),'某种体检异常指标分析'],
    [('询问保单借款','如何申请某项服务'),'如何申请某项服务'],
    [('保障项目的定义','区分疾病种类的标准'),'区分疾病种类的标准'],
    [('询问体检或验尸', '体检的时限要求'),'体检的时限要求'],
    [('投保途径','某渠道的投保流程'),'某渠道的投保流程'],
    [('某产品的最低保额','询问最低保额'),'某产品的最低保额'],
    [('不同保障项目的重复理赔','情景是否赔偿某个保障项目'),'不同保障项目的重复理赔'],
    [('不同保障项目的重复理赔','保障项目的理赔方式'),'不同保障项目的重复理赔'],
    [('询问公司介绍','询问公司定义'),'询问公司介绍'],
    [('询问保险责任','某种疾病是否在承保范围内','某种疾病的保障范围'),'某种疾病的保障范围'],
    [('保障项目的赔付次数','保障项目的赔付','保障项目的理赔方式'),'保障项目的赔付次数'],
    [('保障项目的赔付次数', '保障项目的赔付'), '保障项目的赔付次数'],
    [( '保障项目的赔付', '保障项目的理赔方式'), '保障项目的理赔方式'],
    [('询问保险金给付', '保障项目的赔付次数','保障项目的赔付' ,'询问赔付比例','保障项目的理赔方式'), '保障项目的赔付次数'],
    [('保障项目的赔付次数','保障项目的理赔方式'),'保障项目的赔付次数'],
    [('询问保费垫缴','保费支付渠道'),'保费支付渠道'],
    [('特殊人群投保规则','询问投保规则'),'特殊人群投保规则'],
    [('询问体检或验尸','体检标准'),'体检标准'],
    [('理赔医院','询问合作医院'),'询问合作医院'],
    [('询问保险期间','产品保障期限'),'产品保障期限'],
    [('购买某项保险产品有无优惠','某人购买产品是否有优惠'),'某人购买产品是否有优惠'],
    [('产品推荐','保险是否有购买限制'),'保险是否有购买限制'],
    [('询问体检或验尸','体检指引'),'体检指引'],
    [('询问现金价值','豁免保费后保单是否存在现金价值'),'豁免保费后保单是否存在现金价值'],
    [('询问缴费年期','投保途径','投保渠道的月缴首期'),'投保渠道的月缴首期'],
    [('询问保费垫缴','询问缴费年期'),'询问缴费年期'],
    [('询问缴费年期','投保渠道的月缴首期'),'投保渠道的月缴首期'],
    [('询问保单借款','保单借款的还款期限'),'保单借款的还款期限'],
    [('询问保费垫缴','询问缴费方式','询问变更缴费方式'),'询问变更缴费方式'],
    [('理赔医院','国外医院的要求'),'国外医院的要求'],
    [('特殊人群投保规则','特殊人群投保限额'),'特殊人群投保限额'],
    [('产品保额限制','特殊人群投保限额'),'特殊人群投保限额'],
    [('产品介绍','保险产品的定义'),'产品介绍'],
    [('投保人条件限制','询问投保规则'),'投保人条件限制'],
    [('询问合同终止','合同自动终止的情景'),'合同自动终止的情景'],
    [('询问增值服务内容','某项体检的包含项'),'某项体检的包含项'],
    [('询问增值服务内容','释义项的定义'),'询问增值服务内容'],
    [('询问体检或验尸','询问增值服务内容'),'询问增值服务内容'],
    [('询问缴费年期','询问变更缴费方式'),'询问变更缴费方式'],
    [('询问联系方式','询问增值服务内容'),'询问联系方式'],
    [('理赔的地域限制','理赔申请后出现特殊情况的解决办法'),'理赔申请后出现特殊情况的解决办法'],
    [('查询方式','投资连结保险投资账户介绍'),'投资连结保险投资账户介绍'],
    [('询问体检或验尸','某种体检异常指标定义'),'某种体检异常指标定义'],
    [('险种介绍','保险种类的定义'),'险种介绍'],
    [('某人购买产品是否有优惠','购买某项保险产品有无优惠'),'某人购买产品是否有优惠'],
    [('询问公司介绍','咨询时间'),'咨询时间'],
    [('询问某产品增值服务项目','询问增值服务内容'),'询问某产品增值服务项目'],
    [('某类险种的缴费方式','询问缴费年期'),'某类险种的缴费方式'],
    [('询问增值服务内容','释义项的定义'),'询问增值服务内容'],
    [('询问合同解除','询问犹豫期'),'询问犹豫期'],
    [('体检指引','询问体检或验尸'),'体检指引'],
    [('咨询时间','询问增值服务内容'),'询问增值服务内容'],
    [('购买某项保险产品有无优惠','投保途径'),'购买某项保险产品有无优惠'],
    [('询问缴费方式','询问变更缴费方式'),'询问变更缴费方式'],
    [('险种推荐','保险是否有购买限制'),'保险是否有购买限制'],
    [('询问附加险','某类险种的缴费方式'),'某类险种的缴费方式'],
    [('询问保险金给付','询问赔付比例'),'询问赔付比例'],
    [('询问赔付比例','保障项目的理赔方式'),'询问赔付比例'],
    [('某种疾病的保障范围','询问赔付比例'),'询问赔付比例'],
    [('询问保额累计','特殊人群投保限额'),'询问保额累计'],
    [('询问变更缴费方式','询问合同生效'),'询问变更缴费方式'],
    [('询问保险条款','保障项目的责任免除详'),'保障项目的责任免除详'],
    [('理赔医院','释义项的定义'),'理赔医院'],
    [('询问缴费年期','释义项的定义'),'询问缴费年期'],
    [('疾病种类的定义','询问增值服务内容'),'询问增值服务内容'],
    [('疾病种类的定义','疾病种类治疗费用'),'疾病种类治疗费用'],
    [('投保人与被保险人关系','投保人和被保险人关系	'),'投保人与被保险人关系'],
    [('保费核算','询问产品价格'),'保费核算'],
    [('疾病种类的定义','疾病种类的赔付流程'),'疾病种类的赔付流程'],
    [('释义项的定义','保险产品的定义'),'保险产品的定义'],
    [('疾病种类的定义','区分疾病种类的标准'),'区分疾病种类的标准'],
    [('询问产品价格','询问保险金额'),'询问保险金额'],
    [('保障项目的理赔方式','询问保险金额'),'询问保险金额'],
    [('保障项目的保障范围','保障项目保哪些疾病'),'保障项目保哪些疾病'],
    [('疾病种类的定义','保障项目的赔付次数'),'保障项目的赔付次数'],
    [('投保人条件限制','险种的年龄限制'),'险种的年龄限制'],
    [('询问保险金额','询问保额规定'),'询问保额规定'],
    [('询问公司有哪些产品','产品推荐'),'产品推荐'],
    [('询问投保年龄','询问缴费年期'),'询问缴费年期'],
    [('保障项目的定义','询问赔付比例'),'询问赔付比例'],
    [('不能豁免','可以豁免'),'不能豁免'],
    [('询问附加险','产品介绍'),'产品介绍'],
    [('不同保障项目的重复理赔','多次赔付'),'多次赔付'],
    [('投保人和被保险人关系','特殊人群投保规则'),'特殊人群投保规则'],
    [('保障项目如何申请理赔','保障项目的赔付次数'),'保障项目的赔付次数']
]


def memory_usage_psutil():
    # return the memory usage in MB
    import psutil,os
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

class IntentRe(object):

    def __init__(self):
        self.id_list=[]
        self.jbs = '|'.join(jb)
        self.bxcps = '|'.join(bxcp)
        self.bzxms = '|'.join(bzxm)
        self.bxzls = '|'.join(bxzl)
        self.jbzl = '|'.join(jbzl)
        self.qjs = '|'.join(qj)
        self.sy='|'.join(sy)
        self.bxjs='|'.join(bxj)
        self.yydjs='|'.join(yydj)
        self.jffss='|'.join(jffs)
        self.yys='|'.join(yy)
        self.dds='|'.join(dd)
        self.bqxs='|'.join(bqx)
        self.pts='微信|支付保'
        self.gsmc='中信保诚'
        self.tsrqs='小孩|老人'
        self.fwxms='境外体检高端版1'
        self.tjxms='乳腺X线钼靶检查'


    def get_entity_type(self,sent:str)->list:
        '''
        获取句子中实体的type
        :param sent:
        :return:
        '''
        entity_type_ls=[]
        if re.search('(%s)' % self.dds, sent):
            entity_type_ls.append('Didian')

        if re.search('(%s)' % self.yys, sent):
            entity_type_ls.append('Yiyuan')

        if re.search('(%s)' % self.jffss, sent):
            entity_type_ls.append('Jiaofeifangshi')

        if re.search('(%s)' % self.yydjs, sent):
            entity_type_ls.append('Yiyuandengji')

        if re.search('(%s)' % self.bxjs, sent):
            entity_type_ls.append('Baoxianjin')

        if re.search('(%s)' % self.sy, sent):
            entity_type_ls.append('Shiyi')

        if re.search('(%s)'%self.jbs,sent):
            entity_type_ls.append('Jibing')

        if re.search('(%s)'%self.bxcps,sent):
            entity_type_ls.append('Baoxianchanpin')

        if re.search('(%s)'%self.bzxms,sent):
            entity_type_ls.append('Baozhangxiangmu')

        if re.search('(%s)'%self.bxzls,sent):
            entity_type_ls.append('Jibingzhonglei')

        if re.search('(%s)'%self.qjs,sent):
            entity_type_ls.append('Qingjing')

        return entity_type_ls

    def conflit(self,label_list):
        '''
        解决冲突
        :param label_list:
        :return:
        '''

        label_list=list(label_list)
        if len(label_list)==1:
            return label_list
        else:
            for ele in conflit_list:
                if sum([1 for e in ele[0] if e in label_list])==len(ele[0]):
                    for e in ele[0]:
                        label_list.remove(e)
                    label_list.append(ele[1])

            label_list = list(label_list)
            # label_list = self.tree.conflit_deal(label_list)
            return label_list

    def intent_class(self,sent:str):
        '''
        意图分类功能实现
        :param sent:
        :return:
        '''

        sent_sub = ' |？|。|，|（.*）|\\(.*\\)'
        sent=sent.replace('\n','').strip()
        sent = re.subn(sent_sub, '', sent)[0]
        label_list=[]

        if sent in FAQ_dict:

            ll=FAQ_dict[sent].strip()
            lls=ll.split(' ')
            label_list=[[e,1.0] for e in lls if e not in [' ','']]
            # pass
        else:
            sents = [e for e in jieba.cut(sent)]
            self.sents = sents
            self.jb_sum = 0
            self.bxcp_sum = 0
            self.bxzl_sum = 0
            self.bzxm_sum = 0
            self.qj_sum = 0
            self.jbzl_sum = 0
            self.sy_sum = 0
            self.bqx_sum=0

            for word in sents:
                if word in jb:
                    self.jb_sum += 1
                elif word in bxcp:
                    self.bxcp_sum += 1
                elif word in bxzl:
                    self.bxzl_sum += 1
                elif word in bzxm:
                    self.bzxm_sum += 1
                elif word in qj:
                    self.qj_sum += 1
                elif word in jbzl:
                    self.jbzl_sum += 1
                elif word in sy:
                    self.sy_sum += 1
                elif word in bqx:
                    self.bqx_sum+=1
            ss = jbzl[:]
            ss.extend(jb)
            self.jbzl_jb_sum = sum([1 for e in sents if e in ss])

            ll=self.DD_bqx(sent)
            if ll:
                label_list=[[ll,1.0]]
            else:
                for i in range(300):
                    ll=self.DD(sent,(i+1))
                    if ll not in del_id and ll not in [None,'',' '] and ll in label_dict:
                        ll_=label_dict[ll]
                        if ll_ not in label_list:
                            label_list.append(ll_)

                if not label_list:
                    new_label_list=self.DD_all(sent)
                    if not new_label_list:
                        label_list=['Other']
                    else:
                        label_list=new_label_list

                else:
                    pass
                label_list=self.conflit(label_list)
                label_list=[[e,1.0] for e in label_list if e not in [' ','']]

        return label_list

    def DD_all(self,sent):
        '''
        当意图太细 关系到实体 不能分出 则用上一层类别分出
        :param sent:
        :return:
        '''
        ss=[]
        pattern='申请理赔|理赔流程|理赔.*申请|如何理赔|怎么赔|怎么理赔'
        if re.search(pattern,sent):
            ss=[label_dict['D5'],label_dict['D55'],label_dict['D140']]

        return ss

    def DD(self,sent:str,id:int)->str:
        '''

        :param sent:
        :return:
        '''
        
        #某种疾病是否在承保范围内
        if id == 1:
            pattern='(%s).*(赔偿|赔不赔|赔不赔偿|能否赔偿|会赔吗|可以赔吗|会赔|不会赔|要赔|会赔偿吗|有保障吗|赔钱吗|赔偿吗|能赔吗|有赔付|保不)|' \
                    '小病能报销|(%s)?.*能不能保.*(%s)?|(%s).*(保吗|保不保)'%(self.jbs,self.jbs,self.jbs,self.jbs)
            if re.search(pattern,sent) and self.jb_sum>=1:

                return 'D1'

        if id == 3:
            #保障项目的责任免除详情
            pattern = '(%s|%s).*(除外责任|责任.*豁免|责任免除|什么是责任免除|哪些情况不赔偿|不赔的情况哪些|存在不赔偿的情况吗|有没有不赔偿的情况呢|什么情况下不保|不保有哪些情况|不赔偿|什么情况下不承担|哪些情况是免赔|有什么情况是不能申请理赔的|不保什么|不保什么	|什么情况下免责|有没有责任免除|项责任免除都有什么|责任免除的都不赔吗|责任免除内容|不保什么|不保的内容|' \
                      '有什么不保的|不保的范围|免责条款)|免责内容|天灾人祸怎么办|(天灾战争导致|自然灾害).*(赔不赔|赔付)|责任免除|情况.*不(赔|理赔|赔付)|' \
                      '什么情况不保'%(self.bzxms,self.bxcps)
            if re.search(pattern, sent) or sent in ['除外责任']:
                return 'D3'
        if id == 4:
            #询问合同变更
            pattern = '(变更|更改|更换).{0,2}(合同|保单信息)|合同变更|更改.*身份资料|保全变更|变更保险金额'
            if re.search(pattern, sent):
                return 'D4'

        if id == 6:
            #询问犹豫期
            pattern = '冷静期|犹豫期|犹豫期是什么|犹豫期.{0,3}多久|犹豫期有什么用|犹豫期有什么意义|有没有犹豫期|什么是犹豫期|犹豫期期间可以做什么|犹豫期有什么好处|如何利用犹豫期'
            if re.search(pattern, sent):
                return 'D6'
        if id == 7:
            #询问保险责任
            pattern = '保险责任|什么是保险责任|什么情况赔偿|赔多少钱|有什么情况是能申请理赔的|能保什么|都保哪些内容|都保啥|保险内容|合同保障|保险公司不承担责任|(%s).*(不保)'%self.bxcps
            if re.search(pattern, sent):
                return 'D7'
        if id == 8:
            #合同恢复
            pattern = '复效合同|合同效力.*恢复|合同恢复|怎么恢复合同|如何恢复合同|恢复合同流程|怎样才能恢复合同|想恢复合同|恢复合同需要什么|可以恢复合同吗|恢复合同需要做什么|恢复合同需要提供什么|什么情况下可以恢复合同|什么时候合同效力恢复|合同恢复是什么'
            if re.search(pattern, sent):
                return 'D8'
        # if id == 9:
        #     #保险事故通知
        #     pattern = '保险事故通知|如何通知保险公司|发生事故后.*通知|什么时间通知保险公司|通知保险公司|通知保险公司'
        #     if re.search(pattern, sent):
        #         return 'D9'
        if id == 10:
            #询问保单借款
            pattern = '保单.{1,2}借款|(保险单借款|保单贷款)|保单.*办理.{1,2}贷款|保单借款|用保单向.*借款|申请保单借款|借款不还会|最高可以借多少|借款期限'
            if re.search(pattern, sent):
                return 'D10'
        if id == 11:
            #询问特别约定
            pattern = '特别约定|有.*特别的?说明|有.*附件的?条件|有.*额外的?条款|有.*特别的?地方|有没有特别的?约定|有没有.*附加的?条件|特别约定是什么'
            if re.search(pattern, sent):
                return 'D11'
        if id == 12:
            #合同终止
            pattern = '合同终止|终止合同|结束合同|合同.*终止'
            if re.search(pattern, sent):
                return 'D12'
        if id == 13:
            #询问合同生效
            pattern = '合同生效|保险.*生效|保单.*生效|开始生效|什么时候生效|什么时候承担保险责任|合同生效日|合同.*生效|会生效|合同成立'
            if re.search(pattern, sent):
                return 'D13'
        if id == 14:
            #询问宣告死亡
            pattern = '死亡.*赔|宣告死亡|死亡后|失踪处理|死亡处理|死亡了怎么办|死亡了怎么处理|死亡了找谁|	死亡处理后合同怎么办'
            if re.search(pattern, sent):
                return 'D14'
        if id == 15:
            #询问如实告知
            pattern = '如实告知|没有及时通知保险公司|健康告知.*核保|健康状况要求|投保.*健康告知|补充告知.*规定|(%s).*(健康告知|如实告知)'
            if re.search(pattern, sent):
                return 'D15'
        if id == 16:
            #询问合同解除
            pattern = '合同解除|合同.*解除|解除.*合同|退保|退费|撤单|我要终止'
            if re.search(pattern, sent):
                return 'D16'
        if id == 17:
            #询问保险期间
            pattern = '保障期限|保.*期限|保终身|保障.*年|保险期间|保险期限|保险期|保险的?年限|保多久|保险.*终止|何时生效|合同有效期|保险持续的?时间'
            if re.search(pattern, sent):
                return 'D17'
        if id == 18:
            #询问保费垫缴
            pattern = '垫缴|保费垫交|保险费.*支付|保费.*支付|保费垫缴|保险费的?垫缴|保险费.*垫缴|保险费垫缴|垫缴保险费|缴付保险费|交清保险费|交清保险?费|保费垫缴'
            pattern_no='(月交|月缴)忘记(交|缴)|忘记.*(缴费|交费)|漏交|(缓交|缓缴).*(保险费|保费)|(保费|保险费)(缓交|迟交)|中途忘记(交费|缴费)|忘记交保费|忘记交了|' \
                    '忘记交钱'
            if re.search(pattern, sent) and not re.search(pattern_no,sent):
                return 'D18'
        if id == 19:
            #询问保额
            pattern = '保险金额|能赔偿多少钱|赔偿金额|投保金额'
            if re.search(pattern, sent):
                return 'D19'
        if id == 20:
            #询问受益人
            pattern = '受益人|钱给谁|赔偿金给谁'
            if re.search(pattern, sent) or sent in ['指定']:
                return 'D20'
        if id == 21:
            #询问体检或验尸
            pattern = '体检|验尸|身体检查'
            if re.search(pattern, sent):
                return 'D21'
        if id == 22:
            #询问未归还款项偿还
            pattern = '未归还款项偿还|未归还款项.*偿还|归还的款项|归还的款项|归还款项|未归还款项'
            if re.search(pattern, sent):
                return 'D22'
        if id == 23:
            #询问合同构成
            pattern = '合同构成|合同有.*构成|合同有.*内容|合同有.*文件|合同包含什么|合同中有什么|合同由.*组成|保险合同.{0,3}构成'
            if re.search(pattern, sent):
                return 'D23'
        if id == 24:
            #询问争议处理
            pattern = '争议处理|合同.*争议|发生.*争议|争议.*处理|处理争议|解决争议|意见不一'
            if re.search(pattern, sent):
                return 'D24'
        if id == 25:
            #询问投保年龄
            pattern = '被保险人.*年龄|投保年龄|多大.*投保|年龄限制|\d{1,2}(岁|周岁).*投保'
            if re.search(pattern, sent):
                return 'D25'
        if id == 26:
            #询问适用币种
            pattern = '支持.*币|保费.*人民币|人民币.*支付|[美元|日元|港币|澳元].*支付|适用币种'
            if re.search(pattern, sent):
                return 'D26'
        if id == 27:
            #询问基本保险金额
            pattern = '基本保险金额'
            if re.search(pattern, sent):
                return 'D27'
        if id == 28:
            #询问合同种类
            pattern = '合同.*属于.*类型|合同.{0,2}种类|什么类型.*合同|属于.*类型.*合同|合同保什么|合同.*有.*意义|合同.*有.*作?用' \
                      '|合同.*有.*帮助|保险合同有.*类|保险合同分类|(%s).*种类'%self.bxzls
            if re.search(pattern, sent):
                return 'D28'
        if id == 29:
            #询问等待期
            pattern = '等待期'
            if re.search(pattern, sent):
                return 'D29'
        if id == 30:
            #询问减额缴清
            pattern = '减额缴清|减额交清'
            if re.search(pattern, sent):
                return 'D30'
        if id == 31:
            #询问缴费方式
            pattern = '微信.*支付|支付宝.*支付|微信.*[缴|交]费|支付宝.*[缴|交]费|保费.*支付|缴费.*方式|怎么[缴|交]费|交费.*方式|支付方式|续期缴费规则'
            if re.search(pattern, sent):
                return 'D31'
        if id == 32:
            #询问缴费年期
            pattern = '缴费年期|缴费期|[交|缴].{1,2}年|.{1,2}年[交|缴]|[年|季|月][缴|交]|已缴年费期数'
            pattern_no='(月交|月缴)忘记(交|缴)|忘记.*(缴费|交费)|漏交|(缓交|缓缴).*(保险费|保费)|(保费|保险费)(缓交|迟交)|中途忘记(交费|缴费)|忘记交保费|忘记交了|' \
                    '忘记交钱'
            if re.search(pattern, sent) and not re.search(pattern_no,sent):
                return 'D32'
        if id == 33:
            #服务内容介绍
            pattern = '投保.*售后服务'
            if re.search(pattern, sent):
                return 'D33'
        if id == 34:
            #产品保哪些疾病
            pattern = '(产品|保险|%s)?.*保.*(哪些|什么).*疾?病|(%s).*保(什么|啥|哪些)'%(self.bxcps,self.bxcps)
            pattern_no = '保险费'

            if not re.search(pattern_no, sent) and re.search(pattern, sent) and self.bxcp_sum>=1:
                return 'D34'
        if id == 35:
            #询问公司有哪些产品
            sent=sent.replace('保险责任','保责')
            pattern = '哪些(产品|保险)|介绍.*(产品|保险)|卖什么|有.{1,2}(产品|保险)|罗列.*(产品|保险)|(有没有|卖).{0,3}(%s|%s)'%(self.bxcps,self.bxzls)
            if re.search(pattern, sent):
                return 'D35'
        if id == 36:
            #询问宽限期定义
            pattern = '什么(是|事).*宽限期|宽限期.*(是什么|释义|定义|是?指)|(介绍|解释).*宽限期'
            if re.search(pattern, sent) or sent.replace('\n', '') in ['宽限期', '保险缓交','缓缴保费']:
                return 'D36'

        if id == 37:
            #询问产品优势
            pattern = '为什么要买(%s)?|(%s|保险|产品).*(优势|优点|特点|特色|值得购买)'%(self.bxcps,self.bxcps)
            if re.search(pattern, sent):
                return 'D37'
        # if id == 38:
        #     #属于
        #     pattern = '属于|算|是否属|是不是在|能否|在.*内'
        #     pattern_no = '计算|算起|打算'
        #     if not re.search(pattern_no, sent):
        #         if re.search(pattern, sent):
        #             return 'D38'
        if id == 39:
            #询问免赔额定义
            pattern = '定义 免配额|什么是免赔额|免赔额.*(是什么|释义|定义|是?指)|介绍一?下免赔额|解释一?下免赔额|了解一?下免赔额'
            pattern_1='(%s).{0,3}免赔额'%self.bxcps
            if re.search(pattern, sent) or sent in ['免赔额']:
                return 'D39'
            elif re.search(pattern_1,sent) and self.bxcp_sum>=1:
                return 'D39'
        if id == 40:
            #询问保险费缴纳
            pattern = '保险费.{1,2}支付|保险费.{1,2}交|保险费缴纳|保险费.*缴[清|付]|交付保险费|缴纳?保险费|保费.*扣保?费|申请.*终止续期保费|(怎么样|怎么|怎样).*(交|缴)保险费|' \
                      '保单保费.*多少|交多少.*保费'
            if re.search(pattern, sent):
                return 'D40'
        if id == 42:
            #询问诉讼时效
            pattern = '诉讼时效|诉讼时限'
            if re.search(pattern, sent):
                return 'D42'
        if id == 43:
            #询问联系方式变更
            pattern = '联系方式变更|通讯方式.{1,2}变更|变更联系方式|变更凉席方式|更改联系方式|变更通讯方式|变更联系方式'
            if re.search(pattern, sent):
                return 'D43'
        if id == 44:
            #询问保险金给付
            pattern = '保险金.{1,3}[给|赔]付|[给|赔]付保险金|保险金.{0,2}给付|保险金给付的条件|分配保险金|保险金分配|理赔金.{1,2}怎么处理|' \
                      '(申请|申领|领取).*(%s)|给付'%self.bxjs
            if re.search(pattern, sent):
                return 'D44'
        if id==45:
            #某科目排名全国第一的医院
            pattern = '第一.*医院|最好.*医院|医院.*最[好|强]'
            if re.search(pattern, sent):
                return 'D45'
        if id==46:
            # 询问区别
            pattern='.*[和|与].*区别|.*[和|与].*不同'
            if re.search(pattern, sent):
                return 'D46'
        if id==47:
            #咨询时间
            pattern='服务时间|使用时间|时间限制|限制时间|使用期限|开通时间|什么时候回复|客服.*上班时间'
            if re.search(pattern, sent):
                return 'D47'
        if id==48:
            #患某疾病是否可以投保
            sent=re.subn('(%s|%s)'%(self.bxjs,self.bxcps),'',sent)[0]
            pattern='(%s).*(保|保不保|投保|购?买)|保.*(%s)'%(self.jbs,self.jbs)

            pattern_no='保障范围|保险|通过核保'
            if re.search(pattern, sent) and not re.search(pattern_no,sent) and self.jb_sum>=1:
                return 'D48'
        if id == 49:
            #保单借款
            pattern = '保单.{1,2}借款|保险单借款|保单.*办理.{1,2}贷款|保单借款|用保单向.*借款|申请保单借款|借款不还会|最高可以借多少|借款期限'
            pattern1='.*[日|月|年].*归还'
            if re.search(pattern, sent) and re.search(pattern1,sent):
                return 'D49'

        if id==50:
            #某种情景是否可以投保
            pattern='(%s).*(投保|保|购?买|参加)|(保险|产品).*保(%s)|意外保不保'%(self.qjs,self.qjs)
            # pattern_no='病|癌|瘤'
            if re.search(pattern, sent) and self.qj_sum>=1:
                return 'D50'
        if id==51:
            # 保险种类的定义
            pattern='什么是.*(%s)|(%s).*(是什么|释义|定义|是?指)|(介绍|解释).*(%s)'%(self.bxzls,self.bxzls,self.bxzls)
            if sent in bxzl:
                return 'D51'
            elif re.search(pattern, sent) and self.bxzl_sum>=1:
                return 'D51'
        if id==52:
            #某情景是否在承保范围内
            pattern = '(%s).*(赔偿|赔不赔|赔不赔偿|能否赔偿|会赔吗|可以赔吗|会赔|不会赔|要赔|会赔偿吗|有保障吗|赔钱吗|赔偿吗|能赔吗|有赔付吗|赔吗)'%self.qjs
            if re.search(pattern, sent) and self.qj_sum>=1:
                return 'D52'

        if id==53:
            #疾病种类的定义
            pattern = '什么是(%s)|(%s)是什么|(%s).{0,4}(定义|释义|是?指|名称)'%(self.jbzl,self.jbzl,self.jbzl)
            if sent in jbzl:
                return 'D53'
            elif re.search(pattern, sent) and self.jbzl_sum>=1:
                return 'D53'
        if id==54:
            # 保障项目的定义
            pattern='什么是.*(%s)|(%s).*(是什么|定义|释义|是?指)|(介绍|解释).*(%s)|(%s)的(%s|%s)？?|(%s)意外|(%s)的?(%s)|(%s).*内容'%(self.bzxms,self.bzxms,self.bzxms,self.bxcps,self.bzxms,self.bxjs,self.bzxms,self.bxcps,self.bzxms,self.bzxms)
            if sent in bzxm:
                return 'D54'
            elif re.search(pattern,sent) and self.bzxm_sum>=1:
                return 'D54'
        if id == 55:
            # 得了某类疾病怎样申请理赔
            pattern = '(%s).*理赔.+要交.*[资|材]料|申请.{1,2}理赔|(%s).*(理赔|微理赔|理赔申请|申请理赔|保险金怎么申请|怎么赔偿|保险索赔的流程|保险索赔申请指南|航班延误保险索赔申请的注意事项	|保险索赔|索赔申请理赔|申请保险金|领取赔偿金|理赔流程|保险怎么赔)|' \
                      '(%s).*(理赔)|(%s).*(怎么|如何).{0,3}(理?赔|赔付)'%(self.jbzl,self.jbzl,self.jbzl,self.jbs)

            if re.search(pattern,sent) and self.jbzl_jb_sum>=1:
                return 'D55'

        if id==56:
            #询问失踪处理
            pattern='失踪处理|失踪了?怎么办|失踪了?怎么处理|失踪处理后|失踪'
            if re.search(pattern,sent):
                return 'D56'

        if id==57:
            #疾病种类包含哪些疾病
            pattern='(%s).{0,3}[包含|有].{0,4}(%s)|(%s).{0,3}[包含|有|种类]|哪些疾病.*(属于|是).*(%s)'%(self.jbzl,self.jbs,self.jbzl,self.jbzl)
            pattern_no='区别|不同'

            if re.search(pattern,sent) and not re.search(pattern_no,sent) and self.jbzl_sum>=1:
                return "D57"

        if id==58:
            #某疾病是否属于某疾病种类
            pattern = '(%s).*(属于?|是|算是|算)(%s)'%(self.jbs,self.jbzl)
            pattern_no = '区别|不同'
            if re.search(pattern, sent) and not re.search(pattern_no, sent):
                return "D58"
        if id==59:
            #询问信息误告
            pattern = '报错.*怎么办|年龄误告'
            if re.search(pattern, sent) :
                return "D59"
        if id==60:
            #询问疾病种类包含哪些
            pattern='疾病种类.*(有|包含)'
            if re.search(pattern, sent) :
                return "D60"
        if id==61:
            #某种疾病的预防
            pattern='预防.*(%s)|(%s).*预防'%(self.jbs,self.jbs)
            if re.search(pattern, sent) :
                return "D61"
        if id==62:
            #某种疾病的高发区域
            pattern='(%s).*高发区域|高发区域'%self.jbs
            if re.search(pattern, sent) :
                return "D62"
        if id==63:
            #某种体检异常指标分析
            pattern = '异常.*指标.*分析'
            if re.search(pattern, sent):
                return "D63"
        if id==64:
            #某种体检异常指标定义
            pattern='异常指标是什么|什么是.*异常指标'
            if re.search(pattern, sent):
                return 'D64'
        if id==66:
            #疾病发病原因
            pattern='(%s).*发病原因|(导致|造成).*(%s)'%(self.jbs,self.jbs)
            if re.search(pattern,sent):
                return 'D66'
        if id==68:
            #某平台使用方法
            pattern='(%s).*(怎么用|绑定|使用|操作|流程)'%self.pts
            pattern_no='申请流程'
            if re.search(pattern, sent) and  not re.search(pattern_no,sent):
                return 'D68'
        if id==69:
            #询问合同领取方式
            pattern='合同.*领?[取|拿]'
            if re.search(pattern,sent):
                return 'D69'
        if id==70:
            #投保途径
            pattern='手机.*(投保|购买)|线上.*投保|线下.*投保'
            pattern_no='合同|电子合同'
            if re.search(pattern,sent) and not re.search(pattern_no,sent):
                return 'D70'
        if id==71:
            #投保人和被保险人关系
            sent=re.subn('作为','',sent)[0]
            pattern='(帮|为|给).*(买|投保)'
            if re.search(pattern,sent):
                return 'D71'
        if id==72:
            #区分疾病种类的标准
            pattern = '判断.*(%s).{0,3}(%s)|(%s).{0,3}(%s).{0,3}(区分|区别|不同|标准)'%(self.jbzl,self.jbzl,self.jbzl,self.jbzl)
            if re.search(pattern,sent):
                return 'D72'
        if id==73:
            #保障项目的年龄限制
            pattern='(未成年人|小孩|老人).*(有|购?买).*(%s)|(%s).*年龄.*限制|\d{1,4}(岁|周岁).*(投保|保)'%(self.bzxms,self.bzxms)
            if re.search(pattern,sent) and self.bzxm_sum>=1:
                return 'D73'
        if id == 74:
            #公司经营状况
            pattern='公司.*(经营|赔付|偿还|偿付)+.*(能力|水平|状况)+|(公司).*倒闭|(经营|赔付|偿还|偿付)能力'
            if re.search(pattern,sent):
                return "D74"
        if id==75:
            #投保的职业要求
            pattern='特殊职业.*有要求|职业.*(限制|要求)|被保人.*职业.*要求|职业等级.*能否.*投保|(%s).*(职业要求|职业)|' \
                    '(从事5类职业|五类职业).*投保|哪些职业.*投保'%self.bxcps
            if re.search(pattern,sent):
                return 'D75'
        if id==76:
            #购买某项保险产品有无优惠
            pattern='(有没有|能不能|可不可以|能|有)?.*(便宜|优惠)|投保优惠'
            if re.search(pattern,sent):
                return 'D76'
        if id == 77:
            #某类险种的缴费方式
            pattern='.*险.*费用是.*交|(年|月|季)缴.*(年|月|季)缴.*区别'
            if re.search(pattern,sent):
                return 'D77'
        if id == 78:
            #体检的时限要求
            pattern='体检时限|(半年|去年)+.*(体检结果|体检报告)+'
            if re.search(pattern,sent):
                return 'D78'
        if id == 79:
            #某渠道的投保流程
            pattern='投保流程|(APP)+.*(购买|投保)+|投保方式'
            if re.search(pattern,sent):
                return 'D79'
        if id ==80:
            #保费核算
            pattern = '(保费|保险费|有社保)+.*价格|保费测算|算.{0,3}保费'
            if re.search(pattern,sent):
                return 'D80'
        if id==81:
            #理赔渠道申请流程
            pattern='微理赔|微信理赔流程|理赔路径|哪里理赔|微理赔'
            if re.search(pattern,sent):
                return 'D81'
        if id==82:
            #某产品的最低保额
            pattern='(%s).*最低保(额|费)|(%s).*最(低|少)(保|保险金额|保额)多少'%(self.bxcps,self.bxcps)
            if re.search(pattern,sent):
                return 'D82'
        if id==83:
            #咨询时间
            pattern='时间.{0,2}限制|工作.{0,2}时间|限制.{0,2}时间|时间.*时候'
            if re.search(pattern,sent):
                return 'D83'
        if id==84:
            #理赔后缓缴保险费的处理办法
            pattern='理赔.*后.*(如何|怎么)+处理'
            if re.search(pattern,sent):
                return 'D84'

        if id == 85:
            #申请某保障项目所需资料
            pattern='申请.*(%s).*资料|理赔需要发票'%self.bzxms
            if re.search(pattern,sent):
                return 'D85'
        if id == 86:
            #公司股东构成
            pattern='公司.*(股东|背景)+'
            if re.search(pattern,sent):
                return 'D86'
        if id==87:
            #不同保障项目的重复理赔
            pattern='赔了.*赔|赔付后.*赔|赔.*之后.*能赔'
            if re.search(pattern,sent):
                return 'D87'
        if id==88:
            #询问产品价格
            pattern='(%s|投保).*(价格|多少钱)'%self.bxcps
            if re.search(pattern,sent):
                return 'D88'

        if id==89:
            #询问公司介绍
            pattern='(介绍|了解).*(公?司)|(公司|企业)(介绍|简介|实力|靠谱)|公司(实力|怎么样)|是什么公司|信诚简介|你们是什么(公司|企业)|(%s).*(什么样).*公司|公司产介|(%s).*(介绍|简介|历史|做什么)' \
                    '|(公司|企业).*(名字|全称)|(企业|公司).*叫什么|公司官网地址|官微|官方微信|有多少分公司|(%s).*改名|查当地公司'%(self.gsmc,self.gsmc,self.gsmc)
            pattern_no='(介绍|了解).*公司.*的'
            pattern_1='公司.*做什么|(简介|介绍|哪个|哪家).*(公司)'
            # if re.search(pattern,sent):
            #     print(sent)
            if re.search(pattern,sent) and not re.search(pattern_no,sent):
                return 'D89'

            elif re.search(pattern_1,sent):
                return 'D89'

        if id==90:
            #产品是否属于某一类险
            pattern='(%s).*(属于|是)+.*(%s)'%(self.bxcps,self.bxzls)
            if re.search(pattern,sent):
                return 'D90'
        if id==91:
            #承保区域
            pattern='异地投保_'
            if re.search(pattern,sent):
                return 'D91'
        if id==92:
            #保障项目的赔付次数
            pattern='重复赔付|(%s|%s).*(赔付?次数|赔付?几次|多次赔付|赔付?多少次|赔付次数)|(%s).*赔.*几次'%(self.bzxms,self.jbzl,self.jbzl)
            if re.search(pattern,sent):
                return "D92"

        if id ==93:
            #保费支付渠道
            pattern='(保费|保险费).*怎么支付|信用卡.*(代扣保险费|缴费|扣费|代扣|保费)|能否刷信用卡|信用卡 保费|怎么交钱'
            if re.search(pattern,sent):
                return 'D93'
        if id==94:
            #财务问卷针对的对象
            pattern='财务问卷'
            if re.search(pattern,sent):
                return 'D94'
        if id==95:
            #疾病种类治疗费用
            pattern='(%s).*治疗.{0,2}费用'%self.jbzl
            if re.search(pattern,sent):
                return 'D95'

        if id==96:
            #特殊人群投保规则
            pattern='(%s).*投保规则|\d{1,3}岁以(下|上).*投保|(%s).*(购买|买|投保).*(%s)?'%(self.tsrqs,self.tsrqs,self.bxcps)
            if re.search(pattern,sent):
                return 'D96'

        if id==97:
            #推荐保额
            pattern='保额.*推荐|(选|推荐).*保额'
            if re.search(pattern,sent):
                return 'D97'

        if id==98:
            #核保不通过解决办法
            pattern='核保.*(未|不|没有)通过|人工核保'
            if re.search(pattern,sent):
                return 'D98'

        if id==99:
            #体检标准
            pattern='体检标准|买多少要体检|体检项目.*要求|\d{1,4}岁.*买.*\d{1,5}万.*体检'
            if re.search(pattern,sent):
                return 'D99'
        if id==100:
            #投保所需资料
            pattern='投保.{0,4}资料'
            if re.search(pattern,sent):
                return 'D100'

        if id==101:
            #询问保险条款
            pattern='保险条款|有条款给我看|条款是什么|保单条款|合同条款|有没有条款|条款|跨越保单年度|保证续保|自动续保|' \
                    '治疗.*可以算在.*当中|门诊的报销有要求|同一次事故住院|是绝对免赔额还是相对免赔额'
            if re.search(pattern,sent):
                return 'D101'

        if id==102:
            #理赔医院
            pattern='理赔医院|指定医院|推荐医院|理赔对医院有什么要求|认可的医院|医院要求'
            if re.search(pattern,sent):
                return 'D102'

        if id==103:
            #保障项目的赔付
            pattern='(%s).*(赔?付|给付)|(理赔|赔付).*(%s).*(怎么理赔|缴费)'%(self.bzxms,self.bxjs)
            if re.search(pattern,sent) and self.bzxm_sum>=1:
                return 'D103'

        if id==104:
            #险种介绍
            pattern='(了解|介绍|什么是).*(%s)|(%s)是什么|(%s).*(定义|释义|是?指|介绍)'%(self.bxzls,self.bxzls,self.bxzls)

            if sent in bxzl:
                return 'D104'
            elif re.search(pattern,sent) and self.bxzl_sum>=1:
                return 'D104'
        if id==105:
            #产品推荐
            pattern='有(%s).*(推荐)?|有(%s).*(推荐)|(有没有|有什么|其他).{0,3}(保险|产品)|应该.*购?买.*(保险|产品)|有没有.*(%s).*推荐|有哪些保险推荐'%(self.bxcps,self.bxcps,self.bxcps)
            if re.search(pattern,sent):
                return 'D105'

        if id==106:
            #工具的使用限制
            pattern='在.*可以使用中信银行APP'
            if re.search(pattern,sent):
                return 'D106'

        if id==107:
            #保险类型判断
            pattern='.*型保险'
            if re.search(pattern,sent):
                return 'D107'

        if id==108:
            #诊断报告的领取
            pattern='(诊断证明|诊断报告).*(拿|领取)'
            if re.search(pattern,sent):
                return 'D108'
        if id==109:
            #某人购买产品是否有优惠
            pattern='员工.*购买.*优惠|员工优惠'
            if re.search(pattern,sent):
                return 'D109'
        if id==110:
            #产品升级介绍
            pattern='(%s).*升级|升级.*(%s)|产品升级.*(定义|释义|是?指)|惠康升级|升级保障'%(self.bxcps,self.bxcps)
            if re.search(pattern,sent):
                return 'D110'
        if id==111:
            #体检指引
            pattern='保额.*体检|体检.*保额|体检报告|体检费用|体检规则'

            if re.search(pattern,sent):
                return 'D111'

        if id==112:
            #询问保险金额
            pattern='(%s)?.*(保额|份数).*限制|(%s).*最(高|低).*购?买|保单.*限制|(最高保额|最高能买多少)|投保限额|(保单|银保通出单)?.*限额|(保额|保险金额).*限制|能投保\d{0,3}万|' \
                    '(%s).*(保额|保险金额)'%(self.bxcps,self.bxcps,self.bxcps)
            if re.search(pattern,sent) or sent in ['保险金额']:
                return 'D112'

        if id==113:
            #某情景是否具备购买产品资格
            pattern='(%s).*(可以|能).*购?买.*(%s)|(%s).*(可以|能).*购?买.*保险'%(self.qjs,self.bxcps,self.qjs)
            if re.search(pattern,sent):
                return 'D113'
        if id==114:
            #询问疾病种类的生存期
            # pattern='(%s).*生存期'%self.jbs
            pattern='生存期'
            pattern_1='(%s)'%self.jbzl
            if re.search(pattern,sent) and re.search(pattern_1,sent):
                return 'D114'

        if id==115:
            #理赔的地域限制
            pattern='国外理赔|国(内|外).*(能不能|可以|有没有|还有的|赔).*(理?赔|赔付)|赔付.*(能不能|可以|有没有|有).*地域性?限制|海外.*(能不能|可以|有没有|有).*(理?赔|赔付)|' \
                    '理赔.*(地域|地区).*(要求|限制)'
            if re.search(pattern,sent):
                return 'D115'

        if id==116:
            #保单借款的还款期限
            pattern='保单借款.*\d{1,4}个?(日|年|月).*归?还|保单借款.*(还款期限|归还)'
            if re.search(pattern,sent):
                return 'D116'

        if id==117:
            #特殊人群投保限额
            pattern='(%s).*投保.*限额|(%s).*买多少.*保额|(%s)(保额)|(%s).*购?买(%s).*限额'%(self.tsrqs,self.tsrqs,self.tsrqs,self.tsrqs,self.bxcps)
            if re.search(pattern,sent):
                return 'D117'

        if id==118:
            #权益间的关系
            pattern='保单借款.*后.*申请.*减额缴清|保单借款.*减额缴清.*同时'
            if re.search(pattern,sent):
                return 'D118'
        if id==119:
            #保障项目的保障范围
            pattern='(%s).*保.*(哪些|意外|什么)'%self.bzxms
            if re.search(pattern,sent):
                return 'D119'
        if id==120:
            #产品介绍
            pattern='什么是.*(%s)|(%s).*(是什么|释义|定义|介绍|保单周年日|是啥)|(介绍|解释|了解|是?指).*(%s)|保险用处'%(self.bxcps,self.bxcps,self.bxcps)
            pattern_no='最低保额'
            if sent in bxcp or re.search(pattern,sent) and not re.search(pattern_no,sent):
                return 'D120'
        if id==121:
            #申请纸质合同
            pattern='有.{0,3}纸质.*合同'
            if re.search(pattern,sent) or sent in ['纸质合同']:
                return 'D121'

        if id==122:
            #疾病种类的赔付流程
            pattern='(%s).*(赔付流程|申请赔付|怎么赔付?|如何赔付?)'%self.jbzl
            if sent in ['轻症赔付','重症赔付']:
                return 'D122'
            elif re.search(pattern,sent) and self.jbzl_sum>=1:
                return 'D122'

        if id==123:
            #理赔速度
            pattern='理赔.{0,2}(速度|要多久)|赔款流程.*(速度|慢|快)|理赔时效|多少天理赔|理赔.{0,4}(几|多少)天|申请理赔什么时候可以到账|理赔到账|(理赔|赔款|赔付).{0,4}(快|慢)|赔付时间多久|' \
                    '理赔.*时间.*多久|(几天|多久).*完成.*理赔|理赔天数'
            if re.search(pattern,sent):
                return 'D123'

        if id==124:
            # 产品保障的地域限制
            pattern='异地(保障|投保|理赔|看病)|保障地区|地域限制'
            if re.search(pattern,sent):
                return 'D124'

        if id==125:
            #合同丢失
            pattern='合同丢失|合同不见.*怎么办|合同找不到'
            if re.search(pattern,sent):
                return 'D125'
        if id==126:
            #电子合同介绍
            pattern='什么是.*电子合同|(介绍|解释|了解).*电子合同|电子合同.*(定义|释义|是?指|是什么)|电子合同|电子保单'
            if re.search(pattern,sent):
                return 'D126'

        if id==127:
            #询问_保单_查询
            pattern='保险真假|保单真假|保险.*(真|假)的|有.*(哪些).*保单|查询.*保单|保单.*查询'
            if re.search(pattern,sent):
                return 'D127'

        if id==128:
            #理赔条件
            pattern='理赔条件|怎么样(可以|才能)(理赔|赔付)|出事故都会赔偿吗|达到.*程度.*赔付'
            if re.search(pattern,sent):
                return 'D128'
        if id==129:
            #赔付方式
            pattern='赔付方式|住院费可以保|住院赔不赔|住院报销赔不赔|看病医疗费用.*报销|(医药费|看诊费).*报销'
            if re.search(pattern,sent):
                return 'D129'
        if id==130:
            #投保指引
            pattern='投保指引|计划.*参加|保险怎么买|(怎么|如何)(缴费|交费|买)|购买方式|(在|去)哪里购?买|怎么投保|哪里可以购?买(%s)|(%s)如何购买|(缴费|交费)途径|' \
                    '(如何|哪里)购买(%s)|(%s).*购买途径|哪种方式支付|如何购买|怎么办理购买|如何投保|' \
                    '(%s).*(保全规则|特别规定|核保规则|附加规则|财务规则)'%(self.bxcps,self.bxcps,self.bxcps,self.bxcps,self.bxcps)
            if re.search(pattern,sent):
                return 'D130'

        if id==131:
            #业务办理提交材料
            pattern='办理.*业务.*(保单|合同)原件|(需要|要).*材料'
            if re.search(pattern,sent):
                return 'D131'

        if id==132:
            #投保人与被保险人关系
            pattern='投保人与被保险人关系|投保人.*为哪些人购?买保险|投被保人关系|投保人(和|与)被保险人|隔代投保'
            if re.search(pattern,sent):
                return 'D132'

        if id==133:
            #投保人条件限制
            pattern='什么人.*作为投保人|投保人条件|哪些.{0,4}保险人|(%s)可以(购买|投保)吗|投保人.{0,4}限制|' \
                    '孕妇.*投保|港澳台居民.*投保'
            pattern_no='体检'
            if re.search(pattern,sent) and not re.search(pattern_no,sent):
                return 'D133'

        if id==134:
            #红利申请规定
            pattern='红利领取|领取现金红利|现金红利.*领取|现金领取'
            if re.search(pattern,sent):
                return 'D134'

        if id==135:
            #合同自动终止的情景
            pattern='信托合同.*自动终止'
            if re.search(pattern,sent):
                return 'D135'
        if id==136:
            #理赔报案的目的
            pattern='理赔报案.*(作用|目的|原因)|理赔.*为什么.*报案'
            if re.search(pattern,sent):
                return 'D136'

        if id==137:
            #险种关联选择申请资料
            pattern='险种关联.*资料'
            if re.search(pattern,sent):
                return 'D137'

        if id==138:
            #变更投保人
            pattern='(变更|更改)投保人'
            if re.search(pattern,sent):
                return 'D138'
        if id==139:
            #查询方式
            pattern='价格查询|保单借款利率多少|查询借款利率|红利储存生息的利率'
            if re.search(pattern,sent):
                return 'D139'
        if id==140:
            #得了某个疾病怎样申请理赔
            pattern='(%s).*申请.*理赔'%self.jbs
            if re.search(pattern,sent) and self.jb_sum>=1:
                return "D140"

        if id==141:
            #疾病是否赔某个保障项目
            pattern='(%s)(赔不赔|赔|会不会赔)(%s)'%(self.jbs,self.bzxms)
            ss=jb[:]
            ss.extend(bzxm)
            if re.search(pattern,sent) and sum([1 for e in self.sents if e in ss])>=2:
                return 'D141'

        if id==142:
            #情景是否赔偿某个保障项目
            pattern = '(赔不赔|赔|会不会赔)(%s)' %self.bzxms
            pattern_no='(%s)'%self.jbs
            if re.search(pattern, sent) and not re.search(pattern_no,sent):
                return 'D142'

        if id==143:
            #情景的定义
            pattern='什么是.*(%s)|(%s).*(是什么|释义|定义|是?指)|(介绍|解释|了解).*(%s)'%(self.qjs,self.qjs,self.qjs)
            if sent in qj:
                return 'D143'
            elif re.search(pattern,sent) and self.qj_sum>=1:
                return 'D143'

        if id==144:
            #疾病的定义
            pattern = '什么(是|叫).*(%s)|(%s).{0,2}(是什么|定义|释义|是?指)|(介绍|解释|定义|了解).*(%s)|(%s)的?(%s)' % (self.jbs, self.jbs, self.jbs,self.bxcps,self.jbs)
            if sent in jb:
                return 'D144'
            elif re.search(pattern, sent) and self.jb_sum>=1:
                return 'D144'

        if id==145:
            #询问保险事故通知
            pattern='事故之?后.*保险公司|及时通知保险公司|保险事故.*通知|保险事故.*告知'
            if re.search(pattern,sent):
                return 'D145'
        if id ==146:
            #疾病包含哪些疾病
            pattern = '(%s).{0,3}[包含|有]哪些' % self.jbs
            if re.search(pattern,sent) and self.jb_sum>=1:
                return 'D146'

        if id==147:
            #如何申请某项服务
            pattern='申请.*(%s)'%self.fwxms
            if re.search(pattern,sent):
                return 'D147'

        if id==148:
            #释义项的定义
            pattern = '(什么是|关于).*(%s)|(%s).*(是什么|定义|释义|是?指)|(介绍|解释|了解).*(%s)|(%s).*(意义|包括.*哪些)|对于.*(%s).*要求' \
                       % (self.sy, self.sy, self.sy,self.sy,self.sy)
            if sent in sy:
                return 'D148'
            elif re.search(pattern, sent) and self.sy_sum>=1:
                return 'D148'

        if id==149:
            #询问分公司联系方式
            pattern='分公司.*联系方式|有(那些|什么)分支机构|联系.*分公司'
            if re.search(pattern,sent):
                return 'D149'

        if id==150:
            #询问复效期
            pattern='合同效力恢复|复效期.*(是什么|多久|作用|有什么用|帮助|意义|好处|多少天|时候算起|怎么样|怎样)|(有没有|什么是|利用).{0,3}复效期|(%s|%s).*复效期'%(self.bxcps,self.bxzls)
            if re.search(pattern,sent) or sent in ['复效期','复效']:
                return 'D150'
        if id==151:
            #询问赔付比例
            pattern='最高赔付比例|赔付比例|赔付标准|赔多少|最高赔付.*百分比'
            if re.search(pattern,sent):
                return 'D151'

        if id==152:
            #询问投保规则
            pattern='投保(规则|规定)|投保要求|投保人条件|投保人.*限制|(什么人|哪些).*(作为|是).*(投保人|保险人)|其他规则|保险要求'
            if re.search(pattern,sent):
                return 'D152'

        if id==153:
            #询问保额累计
            pattern='保额累计'
            if re.search(pattern,sent):
                return 'D153'

        if id==154:
            #询问投保人
            pattern='(那些人|哪些人).*(投保|参保)|指定.*投保人'
            if re.search(pattern,sent):
                return 'D154'

        if id==155:
            #询问变更缴费方式
            pattern='(变更|更改).*缴费方式|缴费.{0,2}方式.*(变更|改变|更换|换)|(年|月|季)缴.*(改成|更换).*(年|月|季)缴'
            if re.search(pattern,sent):
                return 'D155'

        if id==300:
            #询问附加险
            pattern='附加.*险|附加AMR'
            if re.search(pattern,sent):
                return 'D156'

        if id==157:
            #询问专科医生
            pattern='专科医生'
            if re.search(pattern,sent):
                return 'D157'

        if id==158:
            #询问现金价值
            pattern='现金价值|现金价格'
            if re.search(pattern,sent):
                return 'D158'
        if id==159:
            #询问体检标准
            pattern='体检标准'
            if re.search(pattern,sent):
                return 'D159'

        if id==160:
            #询问投保书填写
            pattern='投保书填写|投保书'
            if re.search(pattern,sent):
                return 'D160'
        if id==161:
            #询问增值服务内容
            pattern='(服务内容|%s).*(是|包括|包含)(什么|哪些)|重疾绿色就医通道服务|会员级别|(%s)|肠癌检测.*供应商|你们的检测机构有权威吗|海外二次诊疗意见.*服务' \
                    '|重疾的预约挂号费需要客户'%(self.tjxms,self.fwxms)
            if re.search(pattern,sent) or sent in ['检测机构']:
                return 'D161'
        if id==162:
            #询问保险种类
            pattern='属于.*(险种|保险类型)|是什么.*保险类型'
            if re.search(pattern,sent):
                return 'D162'
        if id==163:
            #询问免赔额数值
            pattern='免赔额.*多少'
            if re.search(pattern,sent):
                return 'D163'
        if id==164:
            #某种疾病是否在承保范围内
            pattern='(%s).*(赔偿|赔不赔|赔不赔偿|能否赔偿|会赔吗|可以赔吗|会赔|不会赔|要赔|会赔偿吗|有保障吗|赔钱吗|赔偿吗|能赔吗|有赔付|保不保|保障范围|赔吗)'%self.jbs
            if re.search(pattern,sent) and self.bzxm_sum==0 and self.jb_sum>=1:
                return 'D164'
        if id==165:
            #复效期有无收取滞纳金
            pattern='复效期.*滞纳金'
            if re.search(pattern,sent):
                return 'D165'

        if id==166:
            #减额缴清的影响
            pattern='(减额交清|减额缴清).*影响'
            if re.search(pattern,sent):
                return 'D166'

        if id==167:
            #询问保险金额
            pattern='(增加|减少|共用).{0,4}保额|(%s).*赔付.*多少钱|(%s).*保额'%(self.bxcps,self.bxcps)
            if re.search(pattern,sent):
                return 'D167'

        if id==168:
            #险种的年龄限制
            pattern='(%s).*岁|终身是.*岁|(%s).*年龄.*限制'%(self.bxzls,self.bxzls)
            if re.search(pattern,sent) and self.bxzl_sum>=1:
                return 'D168'

        if id==169:
            #某种疾病是否在承保范围内
            pattern='(%s).*(保障范围|赔付)'%self.jbs
            pattern_no='(%s)'%self.bxjs
            if re.search(pattern,sent) and self.jb_sum>=1:
                return 'D169'
        if id== 170:
            #高发疾病
            pattern='高发疾病|疾病.*高发|常见的疾病有哪些'
            if re.search(pattern,sent):
                return 'D170'
        if id==171:
            #产品保障期限
            pattern='(%s).*(保障终身|保障多久|保障时间|保障期限|保多久|保终身|保障多少年|保多长时间|保险期间|保障期|保障到终老|保障到几岁)|合同到期后.*保障'%self.bxcps
            if re.search(pattern,sent) and self.bxcp_sum>=1:
                return 'D171'
        if id==172:
            #忘记交保费的解决办法
            pattern='(月交|月缴)忘记(交|缴)|忘记.*(缴费|交费)|漏交|(保费|保险费)(缓交|迟交)|中途忘记(交费|缴费)|忘记交保费|忘记交了|' \
                    '忘记交钱'
            if re.search(pattern,sent):
                return 'D172'

        if id==173:
            #保险是否有购买限制
            pattern='(购?买过|有).*(保险|医保|社保).*冲突|其他公司.*买|其他.*(产品|保险).*冲突|' \
                    '买过.*(%s).*再(投保|买).*(%s)|已经买过.*能否.*投保.*(%s)|(%s).*还能买.*(%s)|(%s).*(%s).*一份'%(self.bxcps,self.bxcps,self.bxcps,self.bxcps,self.bxcps,self.bxcps,self.bxcps)

            if re.search(pattern,sent):
                return 'D173'


        if id==174:
            #理赔申请后出现特殊情况的解决办法
            pattern='理赔.*后.*(怎么|需)(理赔|解决|缴费)|国外.*理赔流程|理赔上门服务'
            if re.search(pattern,sent):
                return 'D174'

        if id==175:
            #豁免保费后保单是否存在现金价值
            pattern='豁免保费.*后.*(现金价值|保单贷款)'
            if re.search(pattern,sent):
                return 'D175'

        if id==176:
            #投保渠道的月缴首期
            pattern='手机投保月缴首期|首期.*月费用|首期要交\d{0,2}个月|月交保费首期'
            if re.search(pattern,sent):
                return 'D176'
        if id==177:
            #保障项目给付一半后的处理方法
            pattern='(%s).*一半.*还需?要'%self.bzxms
            if re.search(pattern,sent):
                return 'D177'
        if id==178:
            #特殊人群的理赔方式
            pattern='(%s).*(赔付|理赔)'%self.tsrqs
            if re.search(pattern,sent):
                return 'D178'
        if id==179:
            #国外医院的要求
            pattern='国外医院.*要求|国外理赔医院|国外.*理赔.*医院.*限制|国外理赔条件|国外能赔'
            if re.search(pattern,sent) and not sent.__contains__('理赔条件'):
                return "D179"
        if id==180:
            #保障项目如何申请理赔
            pattern_no='理赔.{0,2}(速度|要多久)|赔款流程.*(速度|慢|快)|理赔时效|多少天理赔|理赔.{0,4}(几|多少)天|申请理赔什么时候可以到账|理赔到账|(理赔|赔款|赔付).{0,4}(快|慢)|赔付时间多久' \
                       '|(%s).*(%s)|免赔额|(%s)的(%s)|理赔后.*收据.*发票.*取回'%(self.bxcps,self.bxjs,self.bxcps,self.bxjs)

            pattern='(%s|%s).*(理?赔|赔偿|赔付)|投保人身故后如何办理|(如何|怎么)理赔'%(self.bzxms,self.bzxms)

            pattern_1 = '(%s).*理赔.+要交.*[资|材]料|申请.{1,2}理赔|(%s).*(怎么理赔|如何进行理赔|可以理赔|理赔路径|微理赔|理赔申请|申请理赔|如何理赔|保后理赔流程|申领应该注意什么|申请理赔|赔偿处理|赔偿要.*材料|保险金申请|赔偿怎么处理|保险金怎么申请|怎么赔偿|保险索赔的流程|保险索赔申请指南|保险索赔|索赔申请理赔|申请保险金|领取赔偿金|理赔流程|保险怎么赔)' \
                      '|(申请|申领).*(%s|%s)|申请.*(医疗费用|津贴理赔).*资料' % (self.bzxms, self.bzxms, self.bxjs, self.bzxms)
            if re.search(pattern_1, sent) and self.bzxm_sum >= 1:
                return 'D180'

            if re.search(pattern,sent) and not re.search(pattern_no,sent) and self.bzxm_sum>=1:
                return 'D180'
        if id==181:
            #投资连结保险投资账户介绍
            pattern='投资连结.*(介绍|有哪些)|产品利率及投连价格查询'
            if re.search(pattern,sent):
                return 'D181'

        if id==182:
            #介绍_定义_产品
            pattern='什么是.*(%s|保险)|(%s).*(是什么|释义|定义|是?指)|(介绍|解释|了解).*(%s)'%(self.bxcps,self.bxcps,self.bxcps)
            pattern_no='最低保额'
            if sent in bxcp:
                return 'D182'
            elif re.search(pattern,sent) and self.bxcp_sum>=1:
                return 'D182'
        if id==183:
            #询问某产品增值服务项目
            pattern='增值服务.*种类|增值服务有什么|(%s).*区别|(%s).*增值服务|增值服务'%(self.fwxms,self.bxcps)
            if re.search(pattern,sent) or sent in ['增值服务']:
                return 'D183'

        if id==184:
            #询问包含产品
            pattern='(%s).*(是|属于).*(产品|保险|公司)'%self.bxcps
            if re.search(pattern,sent) and self.bxcp_sum>=1:
                return 'D184'

        if id==185:
            #询问合作医院
            pattern='(合作医院|合作.*医院).*有哪些|合作医院|第一.*医院|最好.*医院|医院.*最[好|强]'
            if re.search(pattern,sent):
                return 'D185'

        if id==186:
            #某产品的最低保额
            pattern='最低保额|最(低|少)(保|保险金额|保额)多少'
            if re.search(pattern,sent):
                return 'D186'
        if id==187:
            #询问保险金额规定
            pattern='保额规定|保额规则|可选保额|保额.*规定'
            if re.search(pattern,sent):
                return 'D187'

        if id==188:
            #询问宽限期时间
            pattern = '多少天.*宽限期|宽限期.*多少天|(%s).*宽限期|中途缴费停了'%self.bxcps
            if re.search(pattern,sent):
                return "D188"

        if id==189:
            #询问公司定义
            pattern='什么是.*公司|公司.*(是什么|释义|定义|是?指)|(介绍|解释).*公司'
            if re.search(pattern,sent):
                return 'D189'

        if id==190:
            #询问险种免赔额数值
            pattern='(%s).*免赔额.*多少'%self.bxzls
            if re.search(pattern,sent) and self.bxzl_sum>=1:
                return 'D190'

        if id==191:
            #险种推荐
            pattern='有(%s).*(推荐)?|有(%s).*(推荐)|(有没有|有什么|其他).{0,3}(保险|产品)|应该.*购?买.*(保险|产品)'%(self.bxzls,self.bxzls)
            if re.search(pattern,sent):
                return 'D191'

        if id==192:
            #咨询引导
            pattern='专业术语.*(不太懂|明白|理解)'
            if re.search(pattern,sent):
                return 'D192'

        if id ==193:
            #保费支出
            pattern='多少钱.*买'
            if re.search(pattern,sent):
                return 'D193'

        if id==194:
            #公司地址
            pattern='公司地址|客服中心.*在.*地方|客服中心地址'
            if re.search(pattern,sent):
                return 'D194'

        if id==195:
            #询问联系方式
            pattern='客服热线|(客服|公司)电话|云助理电话'
            if re.search(pattern,sent):
                return 'D195'

        if id==196:
            #询问合作银行
            pattern='合作银行|银行购买|银行网点.*购买'
            if re.search(pattern,sent):
                return 'D196'

        if id==197:
            #商业保险与医保的关系
            pattern='医保.*冲突|社保.*需要.*买|(%s).*(医保|社保|医社保).*冲突'%self.bxcps
            if re.search(pattern,sent):
                return 'D197'

        if id==198:
            #保单回溯
            pattern='(办理|申请).*保单回溯|保单回溯|(%s).*回溯'%self.bxcps
            if re.search(pattern,sent) and self.bxcp_sum>=1:
                return 'D198'

        if id==199:
            #某体检的意义
            pattern='(%s).*意义'%(self.tjxms)
            if re.search(pattern,sent):
                return 'D199'
        if id==200:
            #中信银行
            pattern='中信银行'
            if re.search(pattern,sent):
                return 'D200'

        if id==201:
            #询问保险产品
            pattern='NONE'
            if re.search(pattern,sent):
                return 'D201'

        if id==202:
            #保险产品对比
            pattern='(%s).*(与|和).*(%s).*(区别|不同)'%(self.bxcps,self.bxcps)
            if re.search(pattern,sent) and self.bxcp_sum>=1:
                return 'D202'

        if id==203:
            #疾病治疗费用
            pattern='(%s).*治疗.*费用|治疗.*费用.*(%s)'%(self.jbs,self.jbs)
            if re.search(pattern,sent) and self.jb_sum>=1:
                return 'D203'

        if id==204:
            #保费核算
            pattern='保费.{0,2}多少|每年.*交多少钱|多少钱(每|一)年|保费(核算|测算|计算)|(保费|保险费|有社保)+.*价格|保费测算|算.{0,3}保费|保费.{0,4}算|交多少钱'
            if re.search(pattern,sent):
                return 'D204'

        if id==205:
            #增值服务使用次数
            pattern='(%s).*(%s).*(使用次数|能用多少次|次数限制|数量限制)'%(self.bxcps,self.fwxms)
            if re.search(pattern,sent):
                return 'D205'

        if id==206:
            # 增值服务使用时间
            pattern='(%s).*(%s).*(使用时间|开始时间|截止时间|使用时效|有效期)'%(self.bxcps,self.fwxms)
            if  re.search(pattern,sent):
                return 'D206'

        if id==207:
            #增值服务亮点
            pattern='(%s).*(%s).*(亮点|服务亮点|特色|优点|好处|特点)'%(self.bxcps,self.fwxms)
            if re.search(pattern,sent):
                return 'D207'

        if id==208:
            #询问保险责任
            pattern='(%s).*(是否|能否).*(保)|(%s).*(保障内容)|(%s).*(保障.*疾病|有哪些保障|有那些保障)|有哪些疾病.*(%s).*保障范围'%(self.bxcps,self.bxcps,self.bxcps,self.bxcps)
            if re.search(pattern,sent) and self.bxcp_sum>=1:
                return 'D208'
        if id==209:
            #产品续保
            pattern = '(%s).*(续保)' % self.bxcps
            if re.search(pattern, sent) and self.bxcp_sum >= 1:
                return 'D209'

        if id==210:
            #产品期满返还
            pattern = '(%s).*(是否|能否|可以)?.*(期满返还)' % self.bxcps
            if re.search(pattern, sent) and self.bxcp_sum >= 1:
                return 'D210'

        if id==211:
            #产品费用与报销
            pattern='(%s).*(费用|报销).*包含|(%s).*费用'%(self.bxcps,self.bxcps)
            pattern1='(医疗费|恶性肿瘤治疗费).*包含|特定门诊医疗费|报销比例|重症监护室津贴'
            if re.search(pattern,sent) and self.bxcp_sum>=1:
                return 'D211'
            elif re.search(pattern1,sent):
                return 'D211'

        if id==212:
            #询问诉讼时效
            pattern='(%s).*诉讼期'%self.bxcps
            if re.search(pattern,sent) and self.bxcp_sum>=1:
                return 'D212'

        if id==213:
            #变更保险金额
            pattern='(减少|变更|降低|增减).*(保额|保险金额)|(%s).*支持增额'%self.bxcps
            if re.search(pattern,sent):
                return 'D213'

        if id==214:
            #某情景下赔偿
            pattern='(%s).*(全残|身故)'%self.qjs
            if re.search(pattern,sent) and self.qj_sum>=1 and self.bzxm_sum==0:
                return 'D214'

        if id==215:
            #保障项目保哪些疾病
            pattern='(%s).*保.*疾病|(%s).*(包括.*哪些|有哪些|保哪些)|哪些疾?病.*(算|是).*(%s)'%(self.bzxms,self.bzxms,self.bzxms)
            if re.search(pattern,sent) and self.bzxm_sum>=1:
                return 'D215'

        if id==216:
            #询问免责条款
            pattern='免责条款'
            if re.search(pattern,sent):
                return 'D216'

        if id==217:
            #不能豁免
            pattern='(等待期内|没过等待期|投保人).*豁免'
            if re.search(pattern,sent):
                return 'D217'

        if id==218:
            #理赔需要的材料
            pattern='理赔.*(材料|需要什么)'
            if re.search(pattern,sent):
                return 'D218'

        if id==219:
            #可以豁免疾病
            pattern='(等待期外|过了等待期|被保险人).*豁免|豁免'
            if re.search(pattern,sent):
                return 'D219'

        if id==220:
            #免赔额抵扣
            pattern='抵扣.*免赔额|免赔额.*抵扣'
            if re.search(pattern,sent):
                return 'D220'

        if id==221:
            #理赔规则
            pattern='赔付规定|理赔规定'
            if re.search(pattern,sent):
                return 'D221'

        if id==222:
            #多次赔付
            pattern='赔.*(%s).*(能否赔|可以赔|还能赔|是否赔).*(%s)'%(self.jbs,self.jbs)
            if re.search(pattern,sent):
                return 'D222'
        if id==223:
            pattern='正则测试意图'
            if re.search(pattern,sent):
                return 'D223'

        if id==225:
            #询问_公司_区域范围
            pattern='哪些.*(分公司|机构)|有.*(分支机构|分公司)|公司.*在哪|营销服务部.*在哪|哪些.*营销服务部|营销服务部'
            pattern_no='电话|联系方式'
            if re.search(pattern,sent) and not re.search(pattern_no,sent):
                return 'D225'
        if id==226:
            #询问_保单_代理人信息
            pattern='代理人.*(电话|姓名|名字)'
            if re.search(pattern,sent):
                return 'D226'

        if id==227:
            #询问_保单_缴纳
            pattern='保单保费.*多少|交多少.*保费'
            if re.search(pattern,sent):
                return 'D227'






    def DD_bqx(self,sent):
        '''
        保全项 意图
        :param sent:
        :return:
        '''
        sents = [e for e in jieba.cut(sent)]
        sent_bqx=[word for word in sents if word in bqx]
        pattern='(%s).*(规定|需要.*材料|所需材料|怎么申请|申请流程|要怎么办|办理流程)'%self.bqxs
        if re.search(pattern,sent):
            if sent_bqx:
                return sent_bqx[0]+'相关规定'


        pattern='(%s).*(申请时间|时候.*申请|申请.*时间)'%self.bqxs
        if re.search(pattern,sent):
            if sent_bqx:
                return sent_bqx[0]+'申请时间'

        pattern = '(%s)?.*(支持|修改).*(保全|什么)|投保之后可以修改什么|支持哪些保全' % self.bxcps
        if re.search(pattern, sent):
            return '保险产品支持保全项'

        pattern = '(趸|期|年|月)缴.*(%s)?.*(变更通讯资料)|(%s)?.*变更.*通讯资料保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更通讯资料的范围'

        pattern = '(趸|期|年|月)缴.*(%s)?.*(转账授权)|(%s).*转账授权保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品银行转账授权的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(更改个人身份资料)|(%s).*更改个人身份资料' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品更改个人身份资料的范围'

        pattern = '(趸|期|年|月)缴.*(%s)?.*(变更签名)|(%s).*变更签名保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更签名的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更受益人)|(%s).*变更受益人保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更受益人的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更投保人)|(%s).*变更投保人保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更投保人的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(补发保单)|(%s).*补发保单保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品补发保单的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更缴费方式)|(%s).*变更缴费方式保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更缴费方式的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更保险计划)|(%s).*变更保险计划保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更保险计划的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(复效)|(%s).*复效保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品复效的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(减额缴清)|(%s).*减额缴清保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品减额缴清的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(取消承保条件)|(%s).*取消承保条件保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品取消承保条件的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(补充告知)|(%s).*补充告知保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品补充告知的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(保单借款)|(%s).*保单借款保全范围|保单还款保全范围 ' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品保单借款的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(保单还款)|(%s).*保单还款保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品保单还款的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(终止保险合同)|(%s).*终止保险合同保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品终止保险合同的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更通讯资料)|(%s).*变更通讯资料保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更通讯资料的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(银行转账授权)|(%s).*转账授权保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品银行转账授权的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(更改个人身份资料)|(%s).*更改个人身份资料保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品更改个人身份资料的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更签名)|(%s).*变更签名保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更签名的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更受益人)|(%s).*变更受益人保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更受益人的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更投保人)|(%s).*变更投保人保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更投保人的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(保费逾期未付选择)|(%s).*保费逾期未付选择保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品保费逾期未付选择的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(补发保单)|(%s).*补发保单保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品补发保单的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更职业等级)|(%s).*变更职业等级保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更职业等级的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更缴费方式)|(%s).*变更缴费方式保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更缴费方式的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(附约变更)|(%s).*附约变更保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品附约变更的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(复效)|(%s).*复效保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品复效的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(取消承保条件)|(%s).*取消承保条件的范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品取消承保条件的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(补充告知)|(%s).*补充告知保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品补充告知的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(保单借款)|(%s).*保单借款保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品保单借款的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(保单还款)|(%s).*保单还款保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品保单还款的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(满期生存确认)|(%s).*满期生存确认保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品满期生存确认的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(终止保险合同)|(%s).*终止保险合同保全范围|期缴可以支持(%s)终止保险合同|终止保险合同保全范围' % (self.bxcps,self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品终止保险合同的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更通讯资料)|(%s).*变更通讯资料保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更通讯资料的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(银行转账授权)|(%s).*银行转账授权保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品银行转账授权的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(更改个人身份资料)|(%s).*更改个人身份资料保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品更改个人身份资料的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更签名)|(%s).*变更签名保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更签名的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更受益人)|(%s).*变更受益人保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更受益人的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更投保人)|(%s).*变更投保人保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更投保人的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(保费逾期未付选择)|(%s).*保费逾期未付选择保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品保费逾期未付选择的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(补发保单)|(%s).*补发保单保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品补发保单的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更职业等级)|(%s).*变更职业等级保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更职业等级的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(变更缴费方式)|(%s).*变更缴费方式保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品变更缴费方式的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(降低主险保额)|(%s).*降低主险保额保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品降低主险保额的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(复效)|(%s).*复效保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品复效的范围'

        pattern = '(趸|期|年|月)缴.*(%s).*(复效)|(%s).*复效保全范围' % (self.bxcps,self.bxcps)
        if re.search(pattern, sent):
            return '保险产品复效的范围'







if __name__ == '__main__':


    starttime=time.time()
    id=IntentRe()
    # # id.id_nor()
    # # ss={}
    # # for k,v in label_dict.items():
    # #     ss[v]=k
    # # label_list=[v for k,v in label_dict.items()]
    # # id_list=id.id_list
    # # for e in id_list:
    # #     if e not in label_list:
    # #         print(e)
    # # print('*'*120)
    # # for e in label_list:
    # #     if e not in id_list:
    # #         print(e,ss[e])
    #
    #
    # # filer=open('./FAQ_1.txt','r')
    # # fw=open('./writeb1.txt','w')
    # # res=[]
    # # for num,line in enumerate(filer.readlines()):
    # #     # _logger.info(num)
    # #     # if num==1:
    # #     if line not in ['','\n']:
    # #         labels=id.intent_class(line)
    # #         ss=[]
    # #         for label in labels:
    # #             ss.append(label[0])
    # #         new_label=" ".join(ss)
    # #         print(new_label+'\t\t'+line)
    # #
    # #         fw.write(new_label)
    # #         fw.write('\t\t')
    # #         fw.write(line.replace('\n',''))
    # #         fw.write('\n')
    # ss=[e for e in open('./write.txt','r').readlines()]
    # for ele in ss:
    #     label=id.intent_class(ele)
    #     print(ele.replace('\n',''),'\t\t',label)


    while True:
        text=input('输入')
        print([e for e in jieba.cut(text)])
        print(id.intent_class(text))
        # print(id.DD_bqx(text))
    # print(id.intent_class('关于放射疗法'))

    # print(id.get_entity_type("脑血栓"))
    # ss={}
    # for k,v in label_dict.items():
    #     ss[v]=[]
    #
    # print(ss)


    # ss=[e.split('#')[0] for e in open('../write.txt','r').readlines()]
    # ss1=[v for k,v in label_dict.items()]
    # out=[e for e in ss  if e not in ss1]
    # print(out)

    end_time=time.time()

    print(end_time-starttime)