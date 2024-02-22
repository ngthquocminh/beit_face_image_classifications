import os
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import uuid
import pathlib
from beit_main import classify_frame
from PIL import Image


app = Flask(__name__)
cors = CORS(app)

app.config['MAX_CONTENT_PATH'] = 20*1024*1024 


@app.route("/")
def home():
    return "hello"

@app.route("/run", methods=["POST"])
@cross_origin()
def api_beit():
    steam_file = request.files['file']
    name_to_save = str(uuid.uuid4())+pathlib.Path(steam_file.filename).suffix
    steam_file.save(name_to_save)
    image = Image.open(name_to_save).convert("RGB")
    result = classify_frame(image)
    os.remove(name_to_save)
    return {"value": result}


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0')
