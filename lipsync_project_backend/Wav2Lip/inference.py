import argparse
import math
import os
import platform
import subprocess
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

import audio
# from face_detect import face_rect
from models import Wav2Lip

from batch_face import RetinaFace
from time import time

import base64

import redis
redis_conn = redis.Redis()

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


print(f"Current file path: {current_file_path}")

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
                    help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                                default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
             help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--out_height', default=480, type=int,
            help='Output video height. Best results are obtained at 480 or 720')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to run the model on. Options are "cpu" or "cuda". Default is "cpu".')


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    results = []
    pady1, pady2, padx1, padx2 = args.pads

    s = time()

    for image, rect in zip(images, face_rect(images)):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    print('face detect time:', time() - s)

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    return results


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu' # Always use CPU
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    # Load checkpoint and map to CPU
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def main():
    # Ensure temp directory exists
    os.makedirs('temp', exist_ok=True)
    args.img_size = 96

    if os.path.isfile(args.face) and args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        args.static = True

    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (int(args.out_height * aspect_ratio), args.out_height))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print ("Number of frames available for inference: "+str(len(full_frames)))

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        subprocess.check_call([
            "ffmpeg", "-y",
            "-i", args.audio,
            "temp/temp.wav",
        ])
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    s = time()

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)
            # Publish the frame via Redis Pub/Sub
            _, buffer = cv2.imencode('.jpg', f)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            redis_conn.publish('video_frames', jpg_as_text)

    out.release()
    print("wav2lip prediction time:", time() - s)

    try:
        subprocess.check_call([
            'ffmpeg', '-y', '-i', 'temp/result.avi', '-i', 'temp/temp.wav',
            os.path.join(RESULTS_DIR, 'result_voice.mp4')
        ])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print("Processing complete")

model = detector = detector_model = None

def do_load(checkpoint_path):
    global model, detector, detector_model

    model = load_model(checkpoint_path)

    # SFDDetector.load_model(device)
    detector = RetinaFace(gpu_id=0, model_path="Wav2Lip/checkpoints/mobilenet.pth", network="mobilenet",device='cpu')
    # detector = RetinaFace(gpu_id=0, model_path="checkpoints/resnet50.pth", network="resnet50")

    detector_model = detector.model

    print("Models loaded")


face_batch_size = 64 * 8

def face_rect(images):
    num_batches = math.ceil(len(images) / face_batch_size)
    prev_ret = None
    for i in range(num_batches):
        batch = images[i * face_batch_size: (i + 1) * face_batch_size]
        all_faces = detector(batch)  # return faces list of all images
        for faces in all_faces:
            if faces:
                box, landmarks, score = faces[0]
                prev_ret = tuple(map(int, box))
            yield prev_ret


if __name__ == '__main__':
    args = parser.parse_args()
    do_load(args.checkpoint_path)
    main()

# import argparse
# import os
# import cv2
# import numpy as np
# import torch
# from flask_socketio import SocketIO, emit
# from models import Wav2Lip
# from batch_face import RetinaFace
# from time import time
# import base64
# import audio  # Import your audio processing module
# from face_detect import face_rect  # Import your face detection function
# import subprocess

# # Argument parser setup
# parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
# parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)
# parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
# parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
# parser.add_argument('--outfile', type=str, help='Video path to save result', default='results/result_voice.mp4')
# parser.add_argument('--static', type=bool, help='If True, use only the first video frame for inference', default=False)
# parser.add_argument('--fps', type=float, help='FPS for static images (default: 25)', default=25.)
# parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding for face detection (top, bottom, left, right)')
# parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model', default=128)
# parser.add_argument('--resize_factor', default=1, type=int, help='Resize factor for input video resolution')
# parser.add_argument('--out_height', default=480, type=int, help='Output video height')
# parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop region for video (top, bottom, left, right)')
# parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Constant bounding box for face detection')
# parser.add_argument('--rotate', default=False, action='store_true', help='Rotate video by 90 degrees if needed')
# parser.add_argument('--nosmooth', default=False, action='store_true', help='Disable smoothing of face detections')
# parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu or cuda)')

# # Constants
# mel_step_size = 16
# device = 'cpu'

# def get_smoothened_boxes(boxes, T):
#     for i in range(len(boxes)):
#         if i + T > len(boxes):
#             window = boxes[len(boxes) - T:]
#         else:
#             window = boxes[i: i + T]
#         boxes[i] = np.mean(window, axis=0)
#     return boxes

# def face_detect(images, args):
#     results = []
#     pady1, pady2, padx1, padx2 = args.pads

#     for image, rect in zip(images, face_rect(images)):
#         if rect is None:
#             cv2.imwrite('temp/faulty_frame.jpg', image)
#             raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

#         y1 = max(0, rect[1] - pady1)
#         y2 = min(image.shape[0], rect[3] + pady2)
#         x1 = max(0, rect[0] - padx1)
#         x2 = min(image.shape[1], rect[2] + padx2)

#         results.append([x1, y1, x2, y2])

#     boxes = np.array(results)
#     if not args.nosmooth: 
#         boxes = get_smoothened_boxes(boxes, T=5)
#     results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

#     return results

# def datagen(frames, mels, args):
#     img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

#     if args.box[0] == -1:
#         if not args.static:
#             face_det_results = face_detect(frames, args)
#         else:
#             face_det_results = face_detect([frames[0]], args)
#     else:
#         y1, y2, x1, x2 = args.box
#         face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

#     for i, m in enumerate(mels):
#         idx = 0 if args.static else i % len(frames)
#         frame_to_save = frames[idx].copy()
#         face, coords = face_det_results[idx].copy()

#         face = cv2.resize(face, (args.img_size, args.img_size))

#         img_batch.append(face)
#         mel_batch.append(m)
#         frame_batch.append(frame_to_save)
#         coords_batch.append(coords)

#         if len(img_batch) >= args.wav2lip_batch_size:
#             img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

#             img_masked = img_batch.copy()
#             img_masked[:, args.img_size // 2:] = 0

#             img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
#             mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

#             yield img_batch, mel_batch, frame_batch, coords_batch
#             img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

#     if len(img_batch) > 0:
#         img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

#         img_masked = img_batch.copy()
#         img_masked[:, args.img_size // 2:] = 0

#         img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
#         mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

#         yield img_batch, mel_batch, frame_batch, coords_batch

# def main(args):
#     # Load the model
#     model = load_model(args.checkpoint_path)
#     args.img_size = 96

#     if os.path.isfile(args.face) and args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
#         args.static = True

#     if not os.path.isfile(args.face):
#         raise ValueError('--face argument must be a valid path to video/image file')

#     elif args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
#         full_frames = [cv2.imread(args.face)]
#         fps = args.fps

#     else:
#         video_stream = cv2.VideoCapture(args.face)
#         fps = video_stream.get(cv2.CAP_PROP_FPS)

#         full_frames = []
#         while 1:
#             still_reading, frame = video_stream.read()
#             if not still_reading:
#                 video_stream.release()
#                 break

#             aspect_ratio = frame.shape[1] / frame.shape[0]
#             frame = cv2.resize(frame, (int(args.out_height * aspect_ratio), args.out_height))

#             if args.rotate:
#                 frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

#             y1, y2, x1, x2 = args.crop
#             if x2 == -1: x2 = frame.shape[1]
#             if y2 == -1: y2 = frame.shape[0]

#             frame = frame[y1:y2, x1:x2]

#             full_frames.append(frame)

#     if not args.audio.endswith('.wav'):
#         subprocess.check_call([
#             "ffmpeg", "-y",
#             "-i", args.audio,
#             "temp/temp.wav",
#         ])
#         args.audio = 'temp/temp.wav'

#     wav = audio.load_wav(args.audio, 16000)
#     mel = audio.melspectrogram(wav)

#     if np.isnan(mel.reshape(-1)).sum() > 0:
#         raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

#     mel_chunks = []
#     mel_idx_multiplier = 80. / fps
#     i = 0
#     while 1:
#         start_idx = int(i * mel_idx_multiplier)
#         if start_idx + mel_step_size > len(mel[0]):
#             mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
#             break
#         mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
#         i += 1

#     full_frames = full_frames[:len(mel_chunks)]

#     batch_size = args.wav2lip_batch_size
#     gen = datagen(full_frames.copy(), mel_chunks, args)

#     for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
#         img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
#         mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

#         with torch.no_grad():
#             pred = model(mel_batch, img_batch)

#         pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

#         for p, f, c in zip(pred, frames, coords):
#             y1, y2, x1, x2 = c
#             p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

#             f[y1:y2, x1:x2] = p

#             # Convert frame to bytes and send via WebSocket
#             _, buffer = cv2.imencode('.jpg', f)
#             jpg_as_text = base64.b64encode(buffer).decode('utf-8')
#             emit('video_frame', {'frame': jpg_as_text}, namespace='/video')

#     emit('video_done', {'message': 'Video processing complete'}, namespace='/video')

# def load_model(path):
#     model = Wav2Lip()
#     checkpoint = torch.load(path, map_location=device)
#     s = checkpoint["state_dict"]
#     new_s = {}
#     for k, v in s.items():
#         new_s[k.replace('module.', '')] = v
#     model.load_state_dict(new_s)
#     model = model.to(device)
#     return model.eval()

# if __name__ == '__main__':
#     args = parser.parse_args()
#     main(args)
