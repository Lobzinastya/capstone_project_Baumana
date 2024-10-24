from flask import Blueprint, render_template, request
from .forms import InputForm
import pickle
import torch
from models.best_model import FullWordLM_LSTM

main = Blueprint('main', __name__)

#Загрузка модели
best_model = FullWordLM_LSTM(hidden_dim=256, num_classes=4, vocab_size=30004, aggregation_type='max').to('cpu')
save_path = 'models/best_model.pt'
best_model.load_state_dict(torch.load(save_path, weights_only=True, map_location='cpu'))
best_model.eval()
#Загрузка словарей
with open('models/data_for_model.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
word2ind = loaded_data['word2ind']
labels_dict = loaded_data['labels_dict']
label2okpdname = loaded_data['label2okpdname']
label2okpd = {j: i for (i, j) in labels_dict.items()}


@main.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm()
    result = None
    prob=None

    if form.validate_on_submit():
        contract_object = form.contract_object.data
        contract_duration = form.contract_duration.data
        contract_cost = form.contract_cost.data
        label, probs = best_model.predict_one(contract_object, contract_duration, contract_cost, word2ind)
        result = f"ОКПД2 {label2okpd[label]} : {label2okpdname[label]}"
        prob = f"{round(probs[label] *100, 2)}%"
        return render_template('index.html', form=form, result=result, probs=prob)

    return render_template('index.html', form=form, result=result, probs=prob)
