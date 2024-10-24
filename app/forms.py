from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.validators import DataRequired

class InputForm(FlaskForm):
    contract_object = StringField('Объект закупки', validators=[DataRequired()])
    contract_duration = IntegerField('Длительность контракта', validators=[DataRequired()])
    contract_cost = IntegerField('Стоимость контракта', validators=[DataRequired()])
    submit = SubmitField('Классифицировать')
