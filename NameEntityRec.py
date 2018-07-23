from entity_recognition.ner import EntityRecognition




class NameEntityRec(object):


    def __init__(self,config):
        self.ner=EntityRecognition(config_=config)


    def main(self,sent):


        line = sent.replace('\n', '').strip()
        sent = ''
        ll = ''
        index = ''
        line = line.replace('\t\t', '\t').replace('Other', 'other').lower()
        if '\t' in line:
            index = str(line).split('\t')[0].strip().replace(' ', '')
            sent = str(line).split('\t')[1].strip().replace(' ', '')
            ll = str(line).split('\t')[2].strip().replace(' ', '')
        else:
            try:
                index = str(line).split(' ')[0].strip().replace(' ', '')
                sent = str(line).split(' ')[1].strip().replace(' ', '')
                ll = str(line).split(' ')[2].strip().replace(' ', '')
            except Exception as ex:

                print('NameEntityRec error:{}/{}'.format(line,ex))
        ss = self.ner.get_entity([sent])[0]  # 实体识别功能

        sent = ' '.join(ss)
        label = ll
        sent = 'bos' + ' ' + sent + ' ' + 'eos'
        entity = ' '.join(['o'] * len(sent.split(' ')))
        res = str(index) + '\t' + sent + '\t' + entity + '\t' + label

        return sent, res
