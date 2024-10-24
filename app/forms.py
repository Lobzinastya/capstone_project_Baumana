from flask_wtf import FlaskForm
from wtforms import StringField,  FloatField, SubmitField
from wtforms.validators import DataRequired

class InputForm(FlaskForm):
    contract_object = StringField('Объект закупки', validators=[DataRequired()])
    contract_duration = FloatField('Длительность контракта', validators=[DataRequired()])
    contract_cost = FloatField('Стоимость контракта', validators=[DataRequired()])
    submit = SubmitField('Классифицировать')
