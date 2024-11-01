import time
import threading

import cv2
import numpy as np
from flask import Flask, Blueprint, Response, render_template, request, jsonify

from . import tracking
from .plot import plot_landmarks
from .config import config, set_value


main = Blueprint("main", __name__)


def create_app():
    app = Flask(__name__)
    app.register_blueprint(main)

    tracking_thread = threading.Thread(target=tracking.start_tracking)
    tracking_thread.daemon = True
    tracking_thread.start()

    return app


@main.route("/")
def index():
    return render_template("index.html")


@main.route("/video")
def video():
    def get_image():
        while True:
            ret, buffer = cv2.imencode(".jpg", tracking.annotated_image)
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            time.sleep(1 / 30)

    return Response(get_image(), mimetype="multipart/x-mixed-replace; boundary=frame")


@main.route("/plot")
def plot():
    def get_plot():
        while True:
            buffer = plot_landmarks()
            frame = buffer.getvalue()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            time.sleep(1 / 30)

    return Response(get_plot(), mimetype="multipart/x-mixed-replace; boundary=frame")


@main.route("/config", methods=["GET"])
def get_config():

    return render_template("config.html", data=config)


@main.route("/config", methods=["POST"])
def post_config():
    data = request.form
    if data is not None:
        for key, value in data.items():
            print(key, value)
            set_value(config, key, float(value))

    return render_template("config.html", data=config)
