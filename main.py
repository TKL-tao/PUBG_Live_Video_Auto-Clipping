from PIL import Image
from moviepy.editor import VideoFileClip
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import plotly.graph_objects as go
from datetime import timedelta
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_video>")
    sys.exit(1)
video_file_path = sys.argv[1]
video_file_path = str(video_file_path)

start_time0 = time.time()
class MyDataset(Dataset):
    def __init__(self, input_features, input_label=None, features_transform=None, labels_transform=None):
        self.input_features = input_features
        self.input_label = input_label
        self.features_transform = features_transform
        self.labels_transform = labels_transform

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        feature = self.input_features[idx]
        if self.features_transform:
            feature = self.features_transform(feature)

        if self.input_label is not None:
            label = self.input_label[idx]
            if self.labels_transform:
                label = self.labels_transform(label)
        else:
            label = 0  # Used for prediction of test data

        return feature, label

def get_game_status(video_file_path, npy_file_path, model, device, step=1):
    '''
    Input: a video file path
    Output: a .npy file containing the predicted result
    '''
    features_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize the shape of every image
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    game_status = []
    video = VideoFileClip(video_file_path)
    duration = video.duration
    video = video.crop(x1=780, y1=720, x2=1140, y2=760)  # Target Frame

    batch_size = 32
    test_data = []
    start_time = time.time()
    for second in range(0, int(duration), step):
        frame = video.get_frame(second)  # This line costs much computational time
        frame = Image.fromarray(frame)
        test_data.append(frame)

    end_time = time.time()
    print('Time of getting frames: {:.1f} s'.format(end_time - start_time))
    start_time = time.time()
    test_dataset = MyDataset(test_data, None, features_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch_features, _ in test_loader:
            batch_features = batch_features.to(device)
            temp = model(batch_features)
            pred_label = torch.argmax(temp, dim=1)  # shape of pred_label is (batch_size)
            game_status.extend(pred_label.cpu().numpy())
    end_time = time.time()
    print('Time of prediction: {:.1f} s'.format(end_time - start_time))

    np.save(npy_file_path, game_status)

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
model = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(), nn.Linear(512, 3)
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load('model/LFL_Res18.params', map_location=torch.device('cpu')))

# video_file_path = 'E:/Bilibili/BilibiliDownload/LFL_TrainVideos/LFL_2024_1_8_1.mp4'
video_name = video_file_path.split('/')[-1]
npy_file_path = 'Plot_Data/' + video_name.split('.')[0] + '.npy'
get_game_status(video_file_path, npy_file_path, model, device, step=1)
end_time0 = time.time()
print('Total time: {:.1f} s'.format(end_time0 - start_time0))



game_status = np.load(npy_file_path)
print("Game status' length: {}".format(len(game_status)))

def deduplicate_data(game_status):
    i = 0
    while i < len(game_status):
        if game_status[i] == 0:
            i += 1
        else:
            current_status = game_status[i]
            i += 1
            if i < len(game_status):
                next_status = game_status[i]
            else:
                break
            while next_status == current_status:
                game_status[i] = 0
                i += 1
                next_status = game_status[i]
    i = 0
    while i < len(game_status):
        if game_status[i] == 2:
            for temp in range(1, 61):
                if game_status[i + temp] == 2:
                    game_status[i + temp] = 0
            i += 60
        i += 1
                    
    return game_status

game_status = deduplicate_data(game_status)
df = pd.DataFrame({
    'x': np.arange(len(game_status)),
    'x_label' : [str(timedelta(seconds=int(x))) for x in np.arange(len(game_status))],
    'y': [3] * len(game_status),
    'games_status': game_status,
})
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df[df['games_status'] == 0]['x'],
    y=df[df['games_status'] == 0]['y'],
    mode='markers',
    marker=dict(symbol='circle', opacity=1),
    name='Alive'
))
fig.add_trace(go.Scatter(
    x=df[df['games_status'] == 1]['x'],
    y=df[df['games_status'] == 1]['y'],
    mode='markers',
    marker=dict(symbol='diamond', size=22, opacity=0.8, color='orange'),
    name='Kill'
))
fig.add_trace(go.Scatter(
    x=df[df['games_status'] == 2]['x'],
    y=df[df['games_status'] == 2]['y'],
    mode='markers',
    marker=dict(symbol='x', size=22, opacity=0.8, color='red'),
    name='Be killed'
))
fig.update_layout(title='LFL PUBG Game Status',
                  annotations=[dict(x=0.85, xref='paper', y=0.9, yref='paper', showarrow=False,
                                    text='Kill: {}'.format(np.sum(game_status == 1)), align="right",
                                    font=dict(family='Times New Roman', size=24, color='orange')),
                               dict(x=0.85, xref='paper', y=1.0, yref='paper', showarrow=False,
                                    text='Be killed: {}'.format(np.sum(game_status == 2)), align="right",
                                    font=dict(family='Times New Roman', size=24, color='red'))])
fig.update_xaxes(
    title_text='Timeline', tickangle=-45,
    tickfont=dict(family='Times New Roman', size=8, color='black'),
    tickvals=df.loc[df['games_status'] != 0, 'x'],
    ticktext=df.loc[df['games_status'] != 0, 'x_label']
    )
fig.update_yaxes(title_text='', showline=False, showgrid=False, showticklabels=False)
# fig.show()
fig.write_html(npy_file_path[:-4] + '.html')

def clip_video(input_video, output_video_folder, clipped_video_duration=60):
    video = VideoFileClip(input_video)
    duration = video.duration
    for i in range(len(game_status)):
        if game_status[i] == 2:
            start_clip = i-int(clipped_video_duration/2) if i-int(clipped_video_duration/2) > 0 else 0
            end_clip = i+int(clipped_video_duration/2) if i+int(clipped_video_duration/2) < int(duration) else int(duration)
            clipped_video = video.subclip(start_clip, end_clip)
            clipped_video.write_videofile(output_video_folder + '/clipped_video_{}.mp4'.format(i))
    video.close()

output_video_folder = 'clipped_videos'
clip_video(video_file_path, output_video_folder)