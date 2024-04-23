import csv
from flask import Flask, request, jsonify

class SimpleServer:

    def start(self, port):
        self.app = Flask("LoggingServer")

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

        self.app.run(port=port)
