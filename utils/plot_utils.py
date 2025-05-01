import numpy as np
import matplotlib.pyplot as plt


def plot_imp(dataset_name, frame_scores, video_name):

    frame_scores = (frame_scores - np.min(frame_scores)) / (np.max(frame_scores) - np.min(frame_scores) + 1e-6)

    x = np.arange(len(frame_scores), dtype=int)
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(x, frame_scores)
    plt.show()
    plt.savefig(dataset_name + "_frame_weights_with_sm/"+ video_name + ".png")
    plt.close()
