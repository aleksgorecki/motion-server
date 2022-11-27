import matplotlib.pyplot as plt

def plot_motion(response_data, savefile):

    fig, ax = plt.subplots()
    for motion_axis in ("x", "y", "z"):
        axis_values = [float(x) for x in response_data[motion_axis]]
        ax.plot(range(len(axis_values)), axis_values, label=motion_axis)
        
    ax.legend()
    ax.set_ylabel("acceleration m/s^2")
    ax.set_xlabel("sample number")
    fig.savefig(savefile)