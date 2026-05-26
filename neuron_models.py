import numpy as np

# ==========================================
# PART 1: NEURON MODELING MODULE
# ==========================================

def integrate_and_fire(input_current, v_rest=0.0, threshold=1.0, dt=0.1):
    """
    1. Basic Integrate-and-Fire Neuron
    Models a purely capacitive membrane that integrates current until an action potential is fired.
    """
    num_steps = len(input_current)
    membrane_potential = np.full(num_steps, v_rest)
    spike_times = []
    
    for t in range(num_steps - 1):
        if membrane_potential[t] > threshold:
            spike_times.append(t * dt)
            membrane_potential[t] = v_rest
            
        membrane_potential[t + 1] = membrane_potential[t] + (input_current[t] * dt)
        
    return membrane_potential, np.array(spike_times)


def refractory_neuron_model(input_current, v_rest=0.0, threshold=1.0, dt=0.1, t_refractory=0.5):
    """
    2. Integrate-and-Fire with Refractory Period
    Introduces an absolute refractory period preventing immediate consecutive spikes.
    """
    num_steps = len(input_current)
    membrane_potential = np.full(num_steps, v_rest)
    spike_times = []
    refractory_timer = 0
    
    for t in range(num_steps - 1):
        if refractory_timer > 0:
            membrane_potential[t] = v_rest
            refractory_timer -= dt
        elif membrane_potential[t] > threshold:
            spike_times.append(t * dt)
            membrane_potential[t] = v_rest
            refractory_timer = t_refractory
            
        membrane_potential[t + 1] = membrane_potential[t] + (input_current[t] * dt)
        
    return membrane_potential, np.array(spike_times)


def leaky_neuron_model(input_current, v_rest=0.0, threshold=1.0, dt=0.1, t_refractory=0.5, tau_m=5):
    """
    3. Leaky Integrate-and-Fire (LIF)
    Incorporates a leak conductance, modeling the passive decay of membrane potential 
    towards the resting state over the time constant (tau_m).
    """
    num_steps = len(input_current)
    membrane_potential = np.full(num_steps, v_rest)
    spike_times = []
    refractory_timer = 0
    
    for t in range(num_steps - 1):
        if refractory_timer > 0:
            membrane_potential[t] = v_rest
            refractory_timer -= dt
        elif membrane_potential[t] > threshold:
            spike_times.append(t * dt)
            membrane_potential[t] = v_rest
            refractory_timer = t_refractory
            
        # Euler integration with leak term
        dv_dt = (input_current[t] - membrane_potential[t]) / tau_m
        membrane_potential[t + 1] = membrane_potential[t] + (dv_dt * dt)
        
    return membrane_potential, np.array(spike_times)


def adaptive_lif_model(input_current, v_rest=0.0, base_threshold=1.0, dt=0.1, t_refractory=0.5, tau_m=5, threshold_jump=2, tau_threshold=50):
    """
    4. LIF with Dynamic Threshold
    Models spike-frequency adaptation. The action potential threshold increases dynamically 
    after each spike and exponentially decays back to baseline.
    """
    num_steps = len(input_current)
    membrane_potential = np.full(num_steps, v_rest)
    spike_times = []
    refractory_timer = 0
    
    current_threshold = base_threshold
    threshold_history = []
    
    for t in range(num_steps - 1):
        if refractory_timer > 0:
            membrane_potential[t] = v_rest
            refractory_timer -= dt
        elif membrane_potential[t] > current_threshold:
            spike_times.append(t * dt)
            membrane_potential[t] = v_rest
            refractory_timer = t_refractory
            current_threshold += threshold_jump # Dynamic threshold increase
            
        # Euler integration for membrane potential
        dv_dt = (input_current[t] - membrane_potential[t]) / tau_m
        membrane_potential[t + 1] = membrane_potential[t] + (dv_dt * dt)
        
        # Exponential decay of the threshold back to base level
        current_threshold += (base_threshold - current_threshold) / tau_threshold
        threshold_history.append(current_threshold)
        
    # Append final threshold state to match array lengths
    threshold_history.append(current_threshold)
        
    return membrane_potential, np.array(spike_times), np.array(threshold_history)
