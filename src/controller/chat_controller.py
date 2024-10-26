import json

from src.app import app
from src.service.chat_service import *
from flask import request, jsonify, render_template


@app.route("/ballchat" , methods=['POST']) # http://localhost:5000/qooqoo
def get_answer():
    user_input = request.json.get('question')
    print(user_input)
    result = main(user_input)
    print(result)
    return jsonify(result)
