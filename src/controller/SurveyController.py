from flask import request
from numpy.random import choice

from src.app import app
from src.service.SurveyService import *

@app.route("/survey/save" , methods = ['get'])
def index():

    question1 = request.args.get('question1')
    print(question1)
    question2 = request.args.get('question2')
    question3 = request.args.get('question3')
    question4 = request.args.get('question4')
    question5 = request.args.get('question5')
    question6 = request.args.get('question6')
    question7 = request.args.get('question7')
    question8 = request.args.get('question8')
    question9 = request.args.get('question9')
    question10 = request.args.get('question10')

    param = {
        "question1": question1,
        "question2": question2,
        "question3": question3,
        "question4": question4,
        "question5": question5,
        "question6": question6,
        "question7": question7,
        "question8": question8,
        "question9": question9,
        "question10": question10
    }
    print(param)
    data = recommendTest( param )
    return data

