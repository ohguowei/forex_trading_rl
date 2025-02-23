# worker.py

import torch
import torch.nn.functional as F
import numpy as np
import traceback
from threading import Lock

# Adjust these imports to match your actual module names
from simulated_env import SimulatedOandaForexEnv
from models import ActorCritic

def worker(
    worker_id: int,
    global_model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    optimizer_lock: Lock,
    max_steps: int = 20
):
    """
    A worker function for asynchronous training in an A3C-like setup.
    Each worker maintains a local copy of the global model, interacts with
    a simulated trading environment, then periodically updates the global model.
    
    Args:
        worker_id (int): Identifier for logging.
        global_model (ActorCritic): The shared global model across all workers.
        optimizer (torch.optim.Optimizer): Shared optimizer for updating global_model.
        optimizer_lock (Lock): A threading.Lock to prevent concurrent optimizer steps.
        max_steps (int): Number of steps to run in each training episode.
    """

    # Create a local copy of the global model's parameters
    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())
    
    # Instantiate the environment
    # Adjust parameters as needed for your setup
    env = SimulatedOandaForexEnv(
        instrument="EUR_USD",
        units=100,
        granularity="H1",
        candle_count=5000,
        spread=0.0002
    )
    
    # Reset the environment to get initial observation
    state = env.reset()  # shape: (time_window, features) in your code
    decisions = np.zeros((1, 16), dtype=np.float32)  # Example shape, adapt if needed
    
    # Convert them to torch tensors (batch_size=1 assumed)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)   # (1, seq_len, features)
    decisions_t = torch.tensor(decisions, dtype=torch.float32)        # (1, decision_dim)

    step_count = 0
    while step_count < max_steps:
        try:
            # Forward pass on local model
            policy_logits, value = local_model(state_t, decisions_t)  # shapes: (1, num_actions), (1, 1)
            probs = F.softmax(policy_logits, dim=1)                  # (1, num_actions)
            action = torch.multinomial(probs, num_samples=1)         # (1, 1) → pick an action index
            
            # Convert action tensor to int
            action_idx = action.item()
            
            # Step in the environment with that action
            next_state, reward, done, _info = env.step(action_idx)
            
            # Convert reward to tensor
            reward_t = torch.tensor([[reward]], dtype=torch.float32)  # shape (1,1)
            
            # Compute advantage: advantage = (reward - value)
            advantage = reward_t - value  # shape (1,1)
            
            # Policy loss: -log(prob_of_action) * advantage
            # We'll index probs with [0, action_idx], ignoring the batch dimension.
            policy_loss = -torch.log(probs[0, action_idx]) * advantage.detach()
            
            # Value loss: MSE of advantage
            value_loss = advantage.pow(2)
            
            # Total loss
            loss = policy_loss + value_loss
            
            # Check if the loss is valid (not NaN or Inf)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"[Worker {worker_id}] Invalid loss at step {step_count}: {loss.item()}")
                step_count += 1
                continue  # Skip this step if the loss is invalid
            
            # Backprop on local model
            optimizer.zero_grad()
            loss.backward()
            
            # Debugging: Check if gradients are valid
            for name, param in local_model.named_parameters():
                if param.grad is None:
                    print(f"[Worker {worker_id}] Gradient is None for parameter {name} at step {step_count}")
                    # If gradient is None, skip updating this parameter
                    continue
                
            # Copy local gradients
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                if local_param.grad is not None:
                    global_param.grad = local_param.grad.clone()
                else:
                    print(f"[Worker {worker_id}] param has no grad.")
                    pass  # This line is just a placeholder; it does nothing
            
            # Update global model (protected by lock)
            with optimizer_lock:
                # Filter out parameters that have grad=None
                for group in optimizer.param_groups:
                    group["params"] = [p for p in group["params"] if p.grad is not None]
                optimizer.step()


            # Sync local model with global model after update
            local_model.load_state_dict(global_model.state_dict())
            
            # Log info
            print(f"[Worker {worker_id}] Step: {step_count}, Loss: {loss.item():.4f}, Reward: {reward:.4f}")
            
            # Prepare next iteration
            if done or next_state is None:
                # If environment says done or next_state is None, reset
                state = env.reset()
            else:
                state = next_state
            
            # Convert state to torch tensor
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # You may also want to update your 'decisions_t' here if you keep track
            # of historical decisions. E.g. shift a buffer or however you do it:
            # decisions_t = update_decisions(decisions_t, action_idx)
            
            step_count += 1
        
        except Exception as e:
            # This catch ensures if something goes wrong, we log the traceback
            print(f"[Worker {worker_id}] Exception during step: {step_count}", e)
            traceback.print_exc()
            break
    
    print(f"[Worker {worker_id}] Finished training cycle at step {step_count}.")