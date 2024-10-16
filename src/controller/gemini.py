from flask import request ,jsonify
from src.app import app
from src.service.gemini_service import *

# gemini 챗봇
@app.route('/gemini' , methods=['GET'])
def gemini():
    keyword = request.args.get('keyword', type=str)
    return gemini(keyword)
