import numpy as np
import matplotlib.pyplot as plt
import os

# Import the updated, biologically named model
from neuron_models import adaptive_lif_model

# ==========================================
# PART 2: TESTING, VISUALIZATION, & METRICS
# ==========================================

def normalise(x):
    """Normalizes an array to a 0-1 range for plotting."""
    return (x - x.min()) / (x.max() - x.min())

def van_rossum_distance(t0, t1, duration, tau_vr=5, dt=0.1):
    """Calculates the Van Rossum distance between two spike trains."""
    n = int(np.round(duration / dt))
    x0 = np.zeros(n)
    x1 = np.zeros(n)
    
    for x, t in [(x0, t0), (x1, t1)]:
        if len(t) > 0:
            x[np.array(np.round(t / dt), dtype=int)] = 1
        
    nk = int(np.round(3 * tau_vr / dt))
    if 2 * nk + 1 > n:
        nk = (n - 1) // 2
        
    T = np.arange(-nk, nk + 1) * dt
    kernel = np.exp(-T / tau_vr) / tau_vr
    
    for x in [x0, x1]:
        x[:] = np.convolve(x, kernel, 'same')
        
    return np.sqrt(np.sum((x0 - x1)**2 * dt) / tau_vr)

def mean_vr_distance(neuron_spike_times, neuron_spike_idx, input_I, dt=0.1, plot_aprox=True):
    """Simulates the model and calculates the mean Van Rossum distance."""
    duration = input_I.shape[1] * dt
    neuron_spikes, model_spikes = [], []

    if plot_aprox:
        plt.figure(figsize=(10, 6))

    for idx_repeat in range(input_I.shape[0]):
        # "Real" neuron spikes from dataset
        n_spikes = neuron_spike_times[neuron_spike_idx == idx_repeat]
        neuron_spikes.append(n_spikes)

        # Model simulation (Using the updated parameters)
        model_v, m_spikes, _ = adaptive_lif_model(
            input_current=input_I[idx_repeat], 
            v_rest=0.0,             # formerly baseline
            base_threshold=1.0,     # formerly threshold
            dt=dt, 
            t_refractory=0.5,       # formerly r_timer
            tau_m=500,              # formerly tau
            threshold_jump=2,       # formerly ath/thr_i
            tau_threshold=5000      # formerly tau_th/thr_tau
        ) 
        model_spikes.append(m_spikes)

        # Plotting
        if plot_aprox and (idx_repeat < 10):
            plt.scatter(m_spikes, idx_repeat * np.ones(len(m_spikes)), marker='.', 
                        color='xkcd:dark seafoam green', label='Neuron model' if idx_repeat == 0 else None)
            plt.scatter(n_spikes, idx_repeat * np.ones(len(n_spikes)) + 0.2, marker='.', 
                        color='xkcd:purple', label='"Real" neuron' if idx_repeat == 0 else None)

    if plot_aprox:
        plt.title("Model vs. Real Neuron Spikes")
        plt.xlabel('Time (ms)')
        plt.ylabel('Repeat')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Calculate Distance
    d = 0
    for t0, t1 in zip(neuron_spikes, model_spikes):
        d += van_rossum_distance(t0, t1, duration, dt=dt)
    d /= len(neuron_spikes)

    if plot_aprox:
        print(f"\nFinal Mean Van Rossum Distance: {d:.4f}")

    return d


def plot_training_data(train_I, train_v, dt=0.1):
    """Plots the input current and membrane potential from the dataset."""
    repeats, num_time_steps = train_I.shape
    state_t = np.arange(num_time_steps) * dt # in ms

    plt.figure(figsize=(10, 8))
    for idx_repeat in range(min(10, repeats)): # plot max 10 repeats
        plt.plot(state_t, idx_repeat + 0.9 * normalise(train_I[idx_repeat, :]), 
                 color='xkcd:purple', label='Input current' if idx_repeat == 0 else None)
        plt.plot(state_t, idx_repeat + 0.9 * normalise(train_v[idx_repeat, :]), 
                 color='xkcd:dark seafoam green', label='Membrane potential' if idx_repeat == 0 else None)
        
    plt.title("Training Data: Input Current vs Membrane Potential")
    plt.xlabel('Time (ms)')
    plt.ylabel('Repeat index')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    train_i_path = 'Data/train_current.csv'
    train_v_path = 'Data/train_traces.csv'
    train_spikes_path = 'Data/train_spikes.csv'
    
    if not os.path.exists(train_i_path):
        print("Error: Training data not found. Please ensure 'train_current.csv', 'train_traces.csv', and 'train_spikes.csv' are in the 'Data' folder.")
    else:
        print("Loading training dataset...")
        train_I = np.loadtxt(train_i_path) 
        train_v = np.loadtxt(train_v_path) 
        train_spike_times, train_spike_idx = np.loadtxt(train_spikes_path) 

        print("1. Plotting training data traces...")
        plot_training_data(train_I, train_v)

        print("\n2. Running model evaluation and calculating Van Rossum Distance...")
        mean_vr_distance(train_spike_times, train_spike_idx, train_I)
