from app.engine import Engine
from flask import jsonify, render_template, request, Flask

app = Flask(__name__)
engine = Engine()


def start():
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def tarin():
    if not engine.validate_folder():
        return jsonify({'success': False, 'message': 'データフォルダが不正です。'})

    return jsonify({'success': engine.train(), 'message': None})


@app.route('/infer', methods=['POST'])
def infer():
    if request.method == 'POST':
        f = request.files['infer_file']
        answer = engine.infer(f)

        if answer is None:
            response = jsonify({})
            response.status_code = 400
            return response
        else:
            return answer


def _is_jpeg_image(image):
    return True


@app.route('/status')
def status():
    return jsonify(engine.status)


@app.route('/labels')
def labels():
    return jsonify(engine.labels)


@app.route('/find_dataset', methods=['GET'])
def find_dataset():
    dataset = engine.find_dataset()
    return jsonify(dataset)
