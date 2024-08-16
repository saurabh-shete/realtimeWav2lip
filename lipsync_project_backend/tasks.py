# tasks.py
import io
import subprocess
from scipy.io.wavfile import write as wav_write
import os
import redis
# Constants
CHECKPOINT_PATH = "Wav2Lip/checkpoints/wav2lip_gan.pth"
UPLOAD_FOLDER = 'static/uploads'

def process_audio_chunk(audio_data, sample_rate, callback=None):
    print("Processing audio chunk")
    audio_buffer = io.BytesIO()
    wav_write(audio_buffer, sample_rate, audio_data)
    audio_buffer.seek(0)

    # Ensure temp directory exists
    os.makedirs('temp', exist_ok=True)

    # Command to run inference
    command = [
        "ffmpeg", "-i", "pipe:0", "temp/temp.wav"
    ]

    # Convert audio stream
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate(input=audio_buffer.read())

    # Command to run Wav2Lip inference
    command = [
        "python", "Wav2Lip/inference.py",
        "--checkpoint_path", CHECKPOINT_PATH,
        "--face", os.path.join(UPLOAD_FOLDER, 'output.jpg'),
        "--audio", "temp/temp.wav",
        "--resize_factor", "1",
        "--device", "cpu"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    print("Process return code:", process.returncode)
    print("Standard Output:", stdout.decode())
    print("Standard Error:", stderr.decode())
    print("Processing complete")

    # Notify clients that processing is done through callback

    redis_conn = redis.Redis()
    redis_conn.publish('audio_processing', 'Video processing complete')
