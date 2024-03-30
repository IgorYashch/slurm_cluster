from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_job_submit():
    data = request.get_json()

    print("Received the following job description:")
    print(data)

    response = {
        'status': 'success',
        'message': 'Job description received successfully.'
    }
    return jsonify(response), 200

if __name__ == "__main__":
    app.run(port=4567)