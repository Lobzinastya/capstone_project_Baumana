from flask import Blueprint, render_template, request
from .forms import InputForm

main = Blueprint('main', __name__)


@main.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm()
    result = None  # Инициализируем результат как None
    if form.validate_on_submit():  # Проверяем, была ли форма отправлена и валидна ли она
        # Извлекаем данные из формы
        contract_object = form.contract_object.data
        contract_duration = form.contract_duration.data
        contract_cost = form.contract_cost.data

        # Заглушка для модели
        # Здесь можно добавить вашу модель ML для классификации
        result = f"Объект закупки: {contract_object}, Длительность: {contract_duration}, Стоимость: {contract_cost}"

        # Выводим результат
        return render_template('index.html', form=form, result=result)

    return render_template('index.html', form=form, result=result)
