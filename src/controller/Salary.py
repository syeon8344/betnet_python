from flask import request ,jsonify
from src.app import app
from src.service.Playercrawl import *
from src.service.salary_service import *

@app.route("/salary",methods=['GET'])
def getSalary():
    name = request.args.get('name',type=str)
    print(name)

    return predictSalary(name)
