# PUBG_Live_Video_Auto-Clipping
## Description
This is a simple project of automatically  clipping my favorite PUBG live stream moments. The **input** is a mp4 file of PUBG live video, and the **outputs** are the funny 1 minute being killed moments in folder clipped_videos and the player status visualization in folder Plot_Data.

## Necessary python environment
- python 3.8.18
- PIL
- moviepy
- torch
- torchvision
- time
- datetime
- numpy
- pandas
- sys
- plotly

## Usage 
1. Clone this repository to your local directory. For example, `D:/Github_clonespace/PUBG_Live_Video_Auto-Clipping`. 

2. Prepare a PUBG live video of mp4 file. For example, `D:/Download_Videos/PUBG_video.mp4`.

3. Anaconda Prompt
```{bash}
activate your_envs_name
cd D:/Github_clonespace/PUBG_Live_Video_Auto-Clipping
python main.py D:/Download_Videos/PUBG_video.mp4
```

![](Usage_Example/Command_Line_Output.png){width=100%}

## Outputs
### 1 minute being killed moments in folder **clipped_videos**
![](Usage_Example/Video_output.png){width=100%}

### player status visualization in folder **Plot_Data**
![](Usage_Example/Figure_output.png){width=100%}

