from flask import request
from numpy.random import choice

from src.app import app


@app.route("/salary",methods=['GET'])
def getSalary():
    name = request.args.get('name',type=str)
    print(name)
    return name