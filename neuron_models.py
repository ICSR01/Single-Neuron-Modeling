import numpy as np

# ==========================================
# PART 1: NEURON MODELING
# ==========================================

def basic_integrate_and_fire(I, baseline=0.0, threshold=1.0, dt=0.1):
    """1. Basic Integrate-and-Fire Neuron"""
    time_steps = len(I)
    v = np.ones(time_steps) * baseline
    spikes = []
    for time in range(time_steps - 1):
        if v[time] > threshold:
            spikes.append(time * dt)
            v[time] = baseline
        v[time + 1] = v[time] + I[time] * dt
    return v, np.array(spikes)

def refractory_neuron_model(I, baseline=0.0, threshold=1.0, dt=0.1, r_timer=0.5):
    """2. Adding a Refractory Period"""
    time_steps = len(I)
    v = np.ones(time_steps) * baseline
    spikes = []
    r_time = 0
    for time in range(time_steps - 1):
        if r_time > 0:
            v[time] = baseline
            r_time -= dt
        elif v[time] > threshold:
            spikes.append(time * dt)
            v[time] = baseline
            r_time = r_timer
        v[time + 1] = v[time] + I[time] * dt
    return v, np.array(spikes)

def leaky_neuron_model(I, baseline=0.0, threshold=1.0, dt=0.1, r_timer=0.5, tau=10.0):
    """3. Adding a Leak (Leaky Integrate-and-Fire)"""
    time_steps = len(I)
    v = np.ones(time_steps) * baseline
    spikes = []
    r_time = 0
    for time in range(time_steps - 1):
        if r_time > 0:
            v[time] = baseline
            r_time -= dt
        elif v[time] > threshold:
            spikes.append(time * dt)
            v[time] = baseline
            r_time = r_timer
        # Leak term introduced here
        v[time + 1] = v[time] + (dt / tau) * (baseline - v[time]) + I[time] * dt
    return v, np.array(spikes)

def adaptive_leaky_neuron_model(I, baseline=0.0, threshold=1.0, dt=0.1, r_timer=0.5, tau=10.0, ath=1.0, tau_th=10.0):
    """4. Adding a Dynamic/Adaptive Threshold"""
    time_steps = len(I)
    v = np.ones(time_steps) * baseline
    th = np.ones(time_steps) * threshold
    spikes = []
    r_time = 0
    for time in range(time_steps - 1):
        # Dynamic threshold decay
        th[time + 1] = th[time] + (dt / tau_th) * (threshold - th[time])
        
        if r_time > 0:
            v[time] = baseline
            r_time -= dt
        elif v[time] > th[time]:
            spikes.append(time * dt)
            v[time] = baseline
            r_time = r_timer
            # Threshold jumps after a spike
            th[time + 1] = th[time] + ath
            
        v[time + 1] = v[time] + (dt / tau) * (baseline - v[time]) + I[time] * dt
    return v, np.array(spikes), th
