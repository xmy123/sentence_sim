
import torch
from transformers import BertConfig, BertTokenizer, BertModel
from scipy.spatial.distance import cosine

torch.set_grad_enabled(False)


class SentenceSim(object):
    def __init__(self):
        self.model_name_path = "./pretrain_model/"
        self.config = BertConfig.from_pretrained(self.model_name_path, output_hidden_states=True)
        self.model = BertModel.from_pretrained(self.model_name_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name_path)

    def predict_two_sentence_sim(self, sentence_1, sentence_2):
        sentence_1_v = self.__get_sentence_f(sentence_1)
        sentence_2_v = self.__get_sentence_f(sentence_2)
        _sentence_sim = self.__get_sentence_sim(sentence_1_v, sentence_2_v)
        return _sentence_sim

    def __get_sentence_f(self, sentence):
        tokens_pt_sentence = self.tokenizer(sentence, return_tensors="pt")
        sentence_outputs = self.model(**tokens_pt_sentence)
        last_hidden_state = sentence_outputs.last_hidden_state
        sentence_v = torch.mean(last_hidden_state, dim=1)
        return sentence_v

    @staticmethod
    def __get_sentence_sim(sentence_1_v, sentence_2_v):
        cosine_sim_0_1 = 1 - cosine(sentence_1_v[0], sentence_2_v[0])
        return cosine_sim_0_1


if __name__ == '__main__':
    sentence_sim_model = SentenceSim()
    sentence_1 = "公司 2021 年全年实现归母营运利润 1480 亿元，同比增长 6.1%，年化营运 ROE 为 18.9%，同比下降0.6 个百分点"
    sentence_2 = "公司 2020 年全年实现归母营运利润 1480 亿元，同比增长 6.1%，年化营运 ROE 为 18.9%，同比下降0.6 个百分点"
    sim = sentence_sim_model.predict_two_sentence_sim(sentence_1, sentence_2)
    print(sim)


