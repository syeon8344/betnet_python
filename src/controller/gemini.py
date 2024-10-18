from flask import request ,jsonify
from src.app import app
from src.service.gemini_service import *

# gemini 챗봇
@app.route('/gemini' , methods=['GET'])
def gemini():
    print( 1 )
    keyword = request.args.get('keyword', default='', type=str)
    print( keyword )
    response_text = geminiService(keyword)
    return jsonify({'response': response_text})
