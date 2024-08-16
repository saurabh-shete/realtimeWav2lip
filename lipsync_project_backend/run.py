# !wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth' -O 'checkpoints/wav2lip.pth'
# !wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth' -O 'checkpoints/wav2lip_gan.pth'
# !wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/resnet50.pth' -O 'checkpoints/resnet50.pth'
# !wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/mobilenet.pth' -O 'checkpoints/mobilenet.pth'

from flask import Flask, request, jsonify, url_for
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
from redis import Redis
from rq import Queue
from werkzeug.utils import secure_filename
import os
from threading import Thread
import redis
# Import the task function from the tasks module
from tasks import process_audio_chunk
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('socketio')
logger.setLevel(logging.DEBUG)

# Constants
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Initialize Flask
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
socketio = SocketIO(app, cors_allowed_origins="*")  # Ensure async_mode is set

# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup Redis connection
redis_conn = Redis()
q = Queue(connection=redis_conn)

# Check if file is an allowed image type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        url = url_for('static', filename=f'uploads/{filename}', _external=True)
        return jsonify({"message": "Image uploaded successfully", "url": url}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

@socketio.on('audio_data', namespace='/video')
def handle_audio_data(data):
    try:
        audio_data = np.frombuffer(data, dtype=np.uint8)

        if len(audio_data) % 2 != 0:
            audio_data = audio_data[:-1]

        audio_data = audio_data.view(dtype=np.int16)
    except ValueError as e:
        print("Error converting audio data:", e)
        return

    sample_rate = 16000

    # Queue the task in Redis
    job = q.enqueue(process_audio_chunk, audio_data, sample_rate)
    print(f"Task queued: {job.get_id()}")

def listen_to_processing_channel():
    redis_conn = redis.Redis()
    pubsub = redis_conn.pubsub()
    pubsub.subscribe('audio_processing', 'video_frames')

    for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel'].decode()

            if channel == 'video_frames':
                # Emit each frame to the client
                frame_data = message['data'].decode()
                socketio.emit('video_frame', {'frame': frame_data}, namespace='/video')
            elif channel == 'audio_processing':
                # Emit video done message
                socketio.emit('video_done', {'message': message['data'].decode()}, namespace='/video')

# Start listener in a separate thread
Thread(target=listen_to_processing_channel, daemon=True).start()

@socketio.on('video_chunk', namespace='/video')
def handle_video_chunk(data):
    print("Received video chunk")
    emit('video_frame', data, broadcast=True, namespace='/video')

@socketio.on('video_end', namespace='/video')
def handle_video_end(data):
    print("Received video end")
    emit('video_done', data, broadcast=True, namespace='/video')

# Run the application
if __name__ == '__main__':
    socketio.run(app, debug=True)
