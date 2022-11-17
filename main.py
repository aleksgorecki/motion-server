from flask import Flask, request
from matplotlib import pyplot as plt


app = Flask(__name__)


@app.route("/")
def check_server():
    return "OK"


@app.route("/recording", methods=["POST"])
def new_recording():
    data = request.json
    x_axis_values = [float(x) for x in data["x"]]
    y_axis_values = [float(y) for y in data["y"]]
    z_axis_values = [float(z) for z in data["z"]]
    sample_no = list(range(len(x_axis_values)))
    fig, ax = plt.subplots()
    ax.plot(sample_no, x_axis_values)
    ax.plot(sample_no, y_axis_values)
    ax.plot(sample_no, z_axis_values)
    fig.savefig("./last_movement.png")
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
