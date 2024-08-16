import numpy as np
from scipy.io.wavfile import read as wav_read, write as wav_write
from moviepy.editor import concatenate_videoclips, VideoFileClip
import subprocess
import os

# Constants
CHUNK_DURATION_MS = 200
FACE_IMAGE_PATH = "output.png"  # Path to your image
AUDIO_FILE_PATH = "male.wav"  # Path to your full audio file (original used directly)
CHECKPOINT_PATH = "Wav2Lip/checkpoints/wav2lip_gan.pth"  # Path to the checkpoint

# Output directories
INPUT_FOLDER = "input_chunks"
OUTPUT_FOLDER = "output_chunks"
FINAL_FOLDER = "final"

# Create directories if they don't exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FINAL_FOLDER, exist_ok=True)

# Function to split audio into chunks
def split_audio_into_chunks(audio, sample_rate, chunk_duration_ms):
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    return [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]

# Function to process each audio chunk with Wav2Lip
def process_audio_chunk(index, chunk, sample_rate):
    temp_audio_path = os.path.join(INPUT_FOLDER, f"temp_chunk_{index}.wav")
    wav_write(temp_audio_path, sample_rate, chunk)

    output_chunk_video_path = os.path.join(OUTPUT_FOLDER, f"output_chunk_{index}.mp4")
    command = [
        "python", 'Wav2Lip/inference.py',
        "--checkpoint_path", CHECKPOINT_PATH,
        "--face", FACE_IMAGE_PATH,
        "--audio", temp_audio_path,
        "--resize_factor", "1",
        "--outfile", output_chunk_video_path,
        "--device", "cpu"
    ]

    print(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    print("Standard Output:", stdout.decode())
    print("Standard Error:", stderr.decode())

    if not os.path.exists(output_chunk_video_path):
        print(f"Error: output file {output_chunk_video_path} was not created.")
        return None

    return output_chunk_video_path

# Function to combine all video chunks
def combine_video_chunks(video_paths):
    clips = [VideoFileClip(path) for path in video_paths if path is not None]
    final_video = concatenate_videoclips(clips, method='compose')
    final_output_path = os.path.join(FINAL_FOLDER, "output_video.mp4")
    final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")

# Main function
def main():
    sample_rate, audio_data = wav_read(AUDIO_FILE_PATH)

    print(f"Sample rate: {sample_rate}")
    print(f"Audio duration (s): {len(audio_data) / sample_rate}")

    audio_chunks = split_audio_into_chunks(audio_data, sample_rate, CHUNK_DURATION_MS)
    video_chunk_paths = []
    for index, chunk in enumerate(audio_chunks):
        result = process_audio_chunk(index, chunk, sample_rate)
        if result:
            video_chunk_paths.append(result)

    if video_chunk_paths:
        combine_video_chunks(video_chunk_paths)
    else:
        print("No video chunks were created successfully.")

    for path in video_chunk_paths:
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    main()
