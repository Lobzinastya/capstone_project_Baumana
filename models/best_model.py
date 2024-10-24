import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from collections import Counter
from typing import List
# import nltk
# nltk.download('punkt')

# import nltk
# nltk.download('punkt_tab')

import re
import string


class FullWordLM_LSTM(nn.Module):
    def __init__(
            self, hidden_dim: int, vocab_size: int, num_classes: int = 4,
            aggregation_type: str = 'max'
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.numeric_linear = nn.Linear(2, hidden_dim)
        self.projection = nn.Linear(hidden_dim * 2, num_classes)
        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)
        self.aggregation_type = aggregation_type

    def forward(self, input_batch) -> torch.Tensor:
        embeddings = self.embedding(input_batch['input_ids'])  # [batch_size, seq_len, hidden_dim]
        rnn_output, _ = self.rnn(embeddings)  # [batch_size, seq_len, hidden_dim]
        if self.aggregation_type == 'max':
            rnn_output = rnn_output.max(dim=1)[0]  # [batch_size, hidden_dim]
        elif self.aggregation_type == 'mean':
            rnn_output = rnn_output.mean(dim=1)  # [batch_size, hidden_dim]
        else:
            raise ValueError("Invalid aggregation_type")
        text_features = self.dropout(self.text_linear(self.non_lin(rnn_output)))  # [batch_size, hidden_dim]
        numeric_features = torch.stack([input_batch['duration'], input_batch['sum']], dim=1)  # [batch_size, 2]
        numeric_features = self.dropout(self.non_lin(self.numeric_linear(numeric_features)))  # [batch_size, hidden_dim]
        combined_features = torch.cat([text_features, numeric_features], dim=1)  # [batch_size, hidden_dim * 2]
        prediction = self.projection(self.non_lin(combined_features))  # [batch_size, num_classes]
        return prediction

    def predict_one(self, text: str, duration: float, sum_value: float, word2ind: dict, max_len: int = 256) -> str:
        """
        Прогнозирует класс и вероятности для одного примера с использованием текста, duration и sum.
        """
        processed_text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokenized_sentence = [word2ind.get(word, word2ind.get('<unk>')) for word in
                              word_tokenize(processed_text, language="russian")]
        bos_id = word2ind.get('<bos>')
        eos_id = word2ind.get('<eos>')
        tokenized_sentence = [bos_id] + tokenized_sentence
        if len(tokenized_sentence) > max_len - 1:
            tokenized_sentence = tokenized_sentence[:max_len - 1]
        tokenized_sentence = tokenized_sentence + [eos_id]
        input_ids = torch.tensor(tokenized_sentence).unsqueeze(0)
        duration_tensor = torch.tensor([duration], dtype=torch.float32)
        sum_tensor = torch.tensor([sum_value], dtype=torch.float32)
        input_batch = {
            'input_ids': input_ids.to(next(self.parameters()).device),
            'duration': duration_tensor.to(next(self.parameters()).device),
            'sum': sum_tensor.to(next(self.parameters()).device)
        }
        self.eval()
        logits = self.forward(input_batch)
        probabilities = F.softmax(logits, dim=-1).squeeze(0)
        predicted_class = torch.argmax(probabilities).item()
        probabilities = probabilities.tolist()
        return predicted_class, probabilities


# #Параметры взяты из окончания файла 2_Модель_приложение.ipynb из папки с notebook
# # Модель восстанавливаем по известной структуре и весам "best_model.pt"
# best_model = FullWordLM_LSTM(hidden_dim=256, num_classes = 4, vocab_size=30004, aggregation_type='max').to('cpu')
# save_path = r'..\models\best_model.pt'
# best_model.load_state_dict(torch.load(save_path, weights_only=True, map_location='cpu'))
# best_model.eval()
#
# # От pickle восстанавливаем данные словарей для модели
# with open('data_for_model.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)
# word2ind = loaded_data['word2ind']
# labels_dict = loaded_data['labels_dict']
# label2okpdname = loaded_data['label2okpdname']
# label2okpd = {j:i for (i, j) in labels_dict.items()}
#
#
# # text = "Выполнение работ по оказанию услуг"
# # duration = 12.5
# # sum_value = 1500.75
#
# text = 'Текущий ремонт туалета, части инженерных сетей цокольного этажа Медицинского колледжа ФГБОУ ВО БГМУ Минздрава России'
# duration = 14
# sum_value = 5_645_355
#
#
# label, probs = best_model.predict_one(text, duration, sum_value, word2ind)
# print('Классификация: ', label, label2okpd[label], label2okpdname[label])
# print('Вероятности классов: ', probs)

