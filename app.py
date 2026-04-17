from flask import Flask, render_template, request, jsonify
import os
import tempfile
import torch
import shutil
import glob
import json
import numpy as np
import pandas as pd
from torch.utils import data
import subprocess

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
# Vercel-friendly paths (using /tmp/ for writable storage)
tmp_base = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = os.path.join(tmp_base, 'isl_uploads')
app.config['KEYPOINTS_FOLDER'] = os.path.join(tmp_base, 'isl_keypoints')

def ensure_dirs():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['KEYPOINTS_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
label_map = None



def load_model():
    global model, label_map
    from models import Transformer
    from configs import TransformerConfig
    from utils import load_label_map
    
    label_map = load_label_map("include50")
    label_map = dict(zip(label_map.values(), label_map.keys()))
    
    config = TransformerConfig(size="large", max_position_embeddings=256)
    model = Transformer(config=config, n_classes=50)
    model = model.to(device)
    
    ensure_dirs()
    
    model_name = "include50_no_cnn_transformer_large.pth"
    # Try local root first, then fall back to /tmp (Vercel writable)
    model_path = model_name if os.path.exists(model_name) else os.path.join(tmp_base, model_name)
    
    # Check for corrupted model files (incomplete zips)
    if os.path.isfile(model_path):
        import zipfile
        if not zipfile.is_zipfile(model_path):
            print("Found corrupted model file (invalid ZIP), removing it...")
            os.remove(model_path)

    if not os.path.isfile(model_path):
        print(f"Downloading pretrained model to {model_path}...")
        links = {
            "include50_no_cnn_transformer_large.pth": "https://api.wandb.ai/files/abdur-ai4bharat/include50-no-cnn/u7wvdsi2/augs_transformer.pth"
        }
        torch.hub.download_url_to_file(links[model_name], model_path, progress=True)
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Model loaded successfully!")

class KeypointsDataset(data.Dataset):
    def __init__(self, keypoints_dir, max_frame_len=200):
        self.files = sorted(glob.glob(os.path.join(keypoints_dir, "*.json")))
        self.max_frame_len = max_frame_len

    def interpolate(self, arr):
        arr_x = arr[:, :, 0]
        arr_x = pd.DataFrame(arr_x)
        arr_x = arr_x.interpolate(method="linear", limit_direction="both").to_numpy()

        arr_y = arr[:, :, 1]
        arr_y = pd.DataFrame(arr_y)
        arr_y = arr_y.interpolate(method="linear", limit_direction="both").to_numpy()

        arr_x = arr_x * 1920
        arr_y = arr_y * 1080

        return np.stack([arr_x, arr_y], axis=-1)

    def combine_xy(self, x, y):
        x, y = np.array(x), np.array(y)
        _, length = x.shape
        x = x.reshape((-1, length, 1))
        y = y.reshape((-1, length, 1))
        return np.concatenate((x, y), -1).astype(np.float32)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        row = pd.read_json(file_path, typ="series")

        pose = self.combine_xy(row.pose_x, row.pose_y)
        h1 = self.combine_xy(row.hand1_x, row.hand1_y)
        h2 = self.combine_xy(row.hand2_x, row.hand2_y)

        pose = self.interpolate(pose)
        h1 = self.interpolate(h1)
        h2 = self.interpolate(h2)

        df = pd.DataFrame.from_dict({
            "uid": row.uid,
            "pose": pose.tolist(),
            "hand1": h1.tolist(),
            "hand2": h2.tolist(),
        })

        pose = np.array(list(map(np.array, df.pose.values))).reshape(-1, 50).astype(np.float32)
        h1 = np.array(list(map(np.array, df.hand1.values))).reshape(-1, 42).astype(np.float32)
        h2 = np.array(list(map(np.array, df.hand2.values))).reshape(-1, 42).astype(np.float32)
        
        final_data = np.concatenate((pose, h1, h2), -1)
        final_data = np.pad(final_data, ((0, self.max_frame_len - final_data.shape[0]), (0, 0)), "constant")

        return {"uid": row.uid, "data": torch.FloatTensor(final_data)}

def process_live_landmarks(landmarks_list, max_frame_len=169):
    # landmarks_list is a list of {pose: [...], hand1: [...], hand2: [...]}
    pose_points_x, pose_points_y = [], []
    hand1_points_x, hand1_points_y = [], []
    hand2_points_x, hand2_points_y = [], []

    for frame in landmarks_list:
        # Pose: Get first 25 points
        p = frame.get('pose', [])
        px = [lm['x'] for lm in p[:25]] if p else [np.nan] * 25
        py = [lm['y'] for lm in p[:25]] if p else [np.nan] * 25
        # Ensure it is exactly 25
        if len(px) < 25: px.extend([np.nan] * (25 - len(px)))
        if len(py) < 25: py.extend([np.nan] * (25 - len(py)))
        
        # Hand1
        h1 = frame.get('hand1', [])
        h1x = [lm['x'] for lm in h1[:21]] if h1 else [np.nan] * 21
        h1y = [lm['y'] for lm in h1[:21]] if h1 else [np.nan] * 21
        if len(h1x) < 21: h1x.extend([np.nan] * (21 - len(h1x)))
        if len(h1y) < 21: h1y.extend([np.nan] * (21 - len(h1y)))

        # Hand2
        h2 = frame.get('hand2', [])
        h2x = [lm['x'] for lm in h2[:21]] if h2 else [np.nan] * 21
        h2y = [lm['y'] for lm in h2[:21]] if h2 else [np.nan] * 21
        if len(h2x) < 21: h2x.extend([np.nan] * (21 - len(h2x)))
        if len(h2y) < 21: h2y.extend([np.nan] * (21 - len(h2y)))

        pose_points_x.append(px)
        pose_points_y.append(py)
        hand1_points_x.append(h1x)
        hand1_points_y.append(h1y)
        hand2_points_x.append(h2x)
        hand2_points_y.append(h2y)

    # Convert to DataFrames for interpolation
    def interpolate_points(points_list, width_scale=1.0, height_scale=1.0):
        df = pd.DataFrame(points_list)
        df = df.interpolate(method="linear", limit_direction="both").fillna(0)
        arr = df.to_numpy() # (frames, points)
        # Scaled (assuming already flattened in dataset.py logic but we need (frames, features))
        return arr

    pose_x = interpolate_points(pose_points_x) * 1920
    pose_y = interpolate_points(pose_points_y) * 1080
    h1_x = interpolate_points(hand1_points_x) * 1920
    h1_y = interpolate_points(hand1_points_y) * 1080
    h2_x = interpolate_points(hand2_points_x) * 1920
    h2_y = interpolate_points(hand2_points_y) * 1080

    # Combine into (frames, features)
    # Each frame has pose_x[0...24], pose_y[0...24], h1_x[0...20], h1_y[0...20], etc.
    # The order in KeypointsDataset: pose (50), h1 (42), h2 (42)
    
    pose_feat = np.stack([pose_x, pose_y], axis=-1).reshape(len(landmarks_list), -1)
    h1_feat = np.stack([h1_x, h1_y], axis=-1).reshape(len(landmarks_list), -1)
    h2_feat = np.stack([h2_x, h2_y], axis=-1).reshape(len(landmarks_list), -1)
    
    final_data = np.concatenate((pose_feat, h1_feat, h2_feat), axis=-1).astype(np.float32)
    
    # Pad or truncate
    if final_data.shape[0] < max_frame_len:
        final_data = np.pad(final_data, ((0, max_frame_len - final_data.shape[0]), (0, 0)), "constant")
    else:
        final_data = final_data[:max_frame_len]
        
    return torch.FloatTensor(final_data).unsqueeze(0) # (1, max_frame_len, features)

@torch.no_grad()
def predict(video_path):
    global model, label_map
    
    temp_keypoints = os.path.join(app.config['KEYPOINTS_FOLDER'], 'single_video')
    if os.path.exists(temp_keypoints):
        shutil.rmtree(temp_keypoints)
    os.makedirs(temp_keypoints)
    
    process_video(video_path, temp_keypoints)
    
    dataset = KeypointsDataset(temp_keypoints, max_frame_len=169)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    prediction = "No Detection"
    for batch in dataloader:
        input_data = batch["data"].to(device)
        output = model(input_data)
        output = torch.argmax(torch.softmax(output, dim=-1), dim=-1).item()
        prediction = label_map[output]
    
    shutil.rmtree(temp_keypoints)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    global model
    return jsonify({
        'status': 'ready' if model is not None else 'loading',
        'device': str(device)
    })

def lazy_load_model():
    global model
    if model is None:
        print("Lazy loading model...")
        load_model()

@app.route('/predict_live', methods=['POST'])
def handle_predict_live():
    lazy_load_model()
    global model, label_map
    data = request.json
    if not data or 'landmarks' not in data:
        return jsonify({'error': 'No landmarks provided'}), 400
    
    # Verify at least some frames have hands
    hands_present = False
    for frame in data['landmarks']:
        if len(frame.get('hand1', [])) > 0 or len(frame.get('hand2', [])) > 0:
            hands_present = True
            break
            
    if not hands_present:
        return jsonify({'prediction': 'No Detection', 'confidence': 0.0})
    
    try:
        input_tensor = process_live_landmarks(data['landmarks'])
        input_tensor = input_tensor.to(device)
        
        output = model(input_tensor)
        
        if len(output.shape) == 3: # (batch, seq, classes)
            output = torch.max(output, dim=1).values

        probabilities = torch.softmax(output, dim=-1)
        confidence = torch.max(probabilities).item()
        prediction_idx = torch.argmax(probabilities, dim=-1).item()
        prediction = label_map[prediction_idx]
        
        print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        
        return jsonify({'prediction': prediction, 'confidence': confidence})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in predict_live: {e}")
        print(error_trace)
        return jsonify({'error': str(e), 'trace': error_trace}), 500

@app.route('/predict', methods=['POST'])
def handle_predict():
    lazy_load_model()
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, video.filename)
    video.save(video_path)
    
    try:
        prediction = predict(video_path)
        shutil.rmtree(temp_dir)
        return jsonify({'prediction': prediction, 'confidence': 0.95})
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local development, load the model immediately
    load_model()
    # Use the port assigned by the environment (Render) or default to 5000 for local dev
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
