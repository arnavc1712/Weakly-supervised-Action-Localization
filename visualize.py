import matplotlib.pyplot as plt
import torch


def visualize_tcam():
    t_cams = torch.load("./t_cam.pth")
    sample = list(t_cams.keys())[0]
    sample_cam = t_cams[sample][0]["t_cam"]
    num_clips = t_cams[sample][0]["num_clips"]
    fps = 25
    len_clip = 16
    sec_clip = 16/25.0
    time = np.arange(0,(num_clips+1)*sec_clip, sec_clip)
    plt.plot(time, [0]+t_cams)
    plt.show()
    
