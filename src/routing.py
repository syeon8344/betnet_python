from app_start import app


@app.route('/test', methods=['GET'])
def index():
    return "Hello Flask!"
