from flask import Flask, request


app = Flask(__name__)


@app.route("/")
def check_server():
    return "OK"


@app.route("/recording", methods=["POST"])
def new_recording():
    data = request.json
    print(data)
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
