from tkinter import messagebox, simpledialog
import os
import re
import customtkinter as ctk
import gym
from stable_baselines3 import PPO, DQN, A2C, TD3, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import gui_utils
import subprocess
import sys
import tempfile
import json
import time
import signal

# threading and queue removed to keep visualization on main thread

# ------------------ Environments configuration ------------------
# Edit `ENV_MAP` below to change the display names and the gym ids used
# by the application. To add your own custom name, edit this dict and
# restart the app. (No UI editing is required — this keeps control in
# code as you requested.)
ENV_MAP = {
    "Cart Pole (very easy)": "CartPole-v1",
    "Mountain Car (very easy)": "MountainCar-v0",
    "Lunar Lander (medium)": "LunarLander-v2",
    "Acrobot (very easy)": "Acrobot-v0",
    "Pendulum (very easy)": "Pendulum-v0",
    "BipedalWalker (hard)": "BipedalWalker-v2",

    
    
}

# Initialize CustomTkinter
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

# Create the main application window
root = ctk.CTk()
root.title("DQNSuite")
root.geometry("900x700")

# Configure grid on root so things expand
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create a container frame
container = ctk.CTkFrame(root)
container.grid(row=0, column=0, sticky="nsew")

# Make container expandable
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

# Define frames as a dictionary
frames = {}


# Home Screen
home_frame = gui_utils.create_frame("HomeScreen", container, frames)
# Make the home frame layout stretch like other screens
home_frame.grid_rowconfigure(0, weight=0)
home_frame.grid_rowconfigure(1, weight=1)
home_frame.grid_columnconfigure(0, weight=1)

# Top banner and version
VERSION = "0.5"
banner = gui_utils.create_banner(home_frame, "DQNSuite", f"Version {VERSION}")
banner.grid(row=0, column=0, sticky="ew")

# Banner action buttons (changelog/spotlight)
def show_changelog():
    changelog_path = os.path.join(os.getcwd(), "CHANGELOG.md")
    if not os.path.exists(changelog_path):
        messagebox.showinfo("Changelogs", "No changelog found.")
        return

    try:
        with open(changelog_path, 'r') as fh:
            text = fh.read()
    except Exception as e:
        messagebox.showerror("Changelogs", f"Failed to read changelog: {e}")
        return

    win = ctk.CTkToplevel(root)
    win.title("Changelog")
    win.geometry("700x520")
    win.transient(root)

    txt = ctk.CTkTextbox(win, width=660, height=440, font=gui_utils.DEFAULT_TEXT_FONT)
    txt.insert("0.0", text)
    txt.configure(state="disabled")
    txt.grid(row=0, column=0, padx=12, pady=12)

    close = ctk.CTkButton(win, text="Close", command=lambda: win.destroy(), font=gui_utils.DEFAULT_BUTTON_FONT)
    close.grid(row=1, column=0, pady=(0, 12))

chg_btn = ctk.CTkButton(banner.action_frame, text="Changelogs", font=gui_utils.DEFAULT_BUTTON_FONT, command=show_changelog)
chg_btn.grid(row=0, column=0, padx=6, pady=6)

spot_btn = ctk.CTkButton(banner.action_frame, text="Spotlight", font=gui_utils.DEFAULT_BUTTON_FONT,
                        command=lambda: messagebox.showinfo("Spotlight", "Community spotlight: There isn't any spotlight content yet! Submit your agents to our forum!"))
spot_btn.grid(row=0, column=1, padx=6, pady=6)

settings_btn = ctk.CTkButton(banner.action_frame, text="Settings", font=gui_utils.DEFAULT_BUTTON_FONT,
                            command=lambda: gui_utils.switch_frame(frames.get("Settings")))
settings_btn.grid(row=0, column=2, padx=6, pady=6)

# Centered home panel for main navigation
home_panel = ctk.CTkFrame(home_frame)
home_panel.grid(row=1, column=0, sticky="nsew", pady=(24,0))
home_panel.grid_columnconfigure(0, weight=1)

welcome_text = gui_utils.create_text_label(home_panel, "Welcome to DQNSuite — train and evaluate RL agents quickly.", 0)
welcome_text.grid(row=0, column=0, pady=(8, 12))

training_button = ctk.CTkButton(home_panel, text="Training Hub", command=lambda: gui_utils.switch_frame(frames["TrainingHub"]), font=gui_utils.DEFAULT_BUTTON_FONT)
training_button.grid(row=1, column=0, pady=10, ipadx=8, sticky="ew", padx=(40, 40))

eval_button = ctk.CTkButton(home_panel, text="Evaluation Mode", command=lambda: (refresh_model_combo(), gui_utils.switch_frame(frames["Evaluation"])), font=gui_utils.DEFAULT_BUTTON_FONT)
eval_button.grid(row=2, column=0, pady=10, ipadx=8, sticky="ew", padx=(40, 40))

doc_button = ctk.CTkButton(home_panel, text="Documentation", command=lambda: gui_utils.switch_frame(frames["Documentation"]), font=gui_utils.DEFAULT_BUTTON_FONT)
doc_button.grid(row=3, column=0, pady=10, ipadx=8, sticky="ew", padx=(40, 40))

about_button = ctk.CTkButton(home_panel, text="About", command=lambda: gui_utils.switch_frame(frames["AboutPage"]), font=gui_utils.DEFAULT_BUTTON_FONT)
about_button.grid(row=4, column=0, pady=10, ipadx=8, sticky="ew", padx=(40, 40))

# Add Evaluation to Home

def refresh_model_combo():
    """Refresh the model file list for the Evaluation screen combo.

    This function is defined early but executed when the button is
    clicked (after `model_combo` has been created), so it's safe to
    reference `model_combo` lazily.
    """
    try:
        models_dir_local = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir_local, exist_ok=True)
        files_local = sorted([f for f in os.listdir(models_dir_local) if f.endswith('.zip')])
        # model_combo is created later; configure it if present
        try:
            model_combo.configure(values=files_local)
            if files_local:
                model_combo.set(files_local[0])
        except Exception:
            pass
    except Exception:
        pass

# Training Hub
training_frame = gui_utils.create_frame("TrainingHub", container, frames)
title_label = gui_utils.create_title_label(training_frame, "Training Parameters")
env_label = gui_utils.create_text_label(training_frame, "Select Environment:", pady=5)
env_var = ctk.CTkComboBox(training_frame, values=[], state="readonly", font=gui_utils.DEFAULT_TEXT_FONT)
# populate combobox with display names from env_list
env_display_names = list(ENV_MAP.keys())
env_var.configure(values=env_display_names)
if env_display_names:
    env_var.set(env_display_names[0])

algo_label = gui_utils.create_text_label(training_frame, "Select Algorithm:", pady=5)
algo_var = ctk.CTkComboBox(training_frame, values=["PPO", "DQN", "A2C", "TD3", "DDPG", "SAC"], state="readonly", font=gui_utils.DEFAULT_TEXT_FONT)
algo_var.set("PPO")

# Add sliders for hyperparameters (each returns a frame to place)
lr_slider = gui_utils.create_hyperparameter_slider(training_frame, "Learning Rate:", 0.0001, 0.01, 0.0001, 0.001)
gamma_slider = gui_utils.create_hyperparameter_slider(training_frame, "Gamma:", 0.9, 0.9999, 0.0001, 0.99)
batch_size_slider = gui_utils.create_hyperparameter_slider(training_frame, "Batch Size:", 8, 256, 1, 64)
buffer_size_slider = gui_utils.create_hyperparameter_slider(training_frame, "Buffer Size:", 1000, 100000, 1, 10000)
epsilon_slider = gui_utils.create_hyperparameter_slider(training_frame, "Epsilon (DQN):", 0.01, 1.0, 0.01, 0.1)
timesteps_slider = gui_utils.create_hyperparameter_slider(training_frame, "Number of Timesteps:", 1000, 1000000, 1, 10000)

# Place training widgets using grid for responsiveness
training_frame.grid_columnconfigure(0, weight=1)
title_label.grid(row=0, column=0, pady=(20, 10))
env_label.grid(row=1, column=0, sticky="w", padx=12)
env_var.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
algo_label.grid(row=3, column=0, sticky="w", padx=12)
algo_var.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 8))

# Sliders frames
lr_slider.grid(row=5, column=0, sticky="ew", padx=12, pady=6)
gamma_slider.grid(row=6, column=0, sticky="ew", padx=12, pady=6)
batch_size_slider.grid(row=7, column=0, sticky="ew", padx=12, pady=6)
buffer_size_slider.grid(row=8, column=0, sticky="ew", padx=12, pady=6)
epsilon_slider.grid(row=9, column=0, sticky="ew", padx=12, pady=6)
timesteps_slider.grid(row=10, column=0, sticky="ew", padx=12, pady=6)


def start_training() -> None:
    env_name = env_var.get()
    algo_name = algo_var.get()
    learning_rate = lr_slider.get()
    gamma = gamma_slider.get()
    batch_size = int(batch_size_slider.get())
    buffer_size = int(buffer_size_slider.get())
    epsilon = epsilon_slider.get()
    num_steps = int(timesteps_slider.get())

    # Map display name to gym id using ENV_MAP
    env_id = ENV_MAP.get(env_name, env_name)

    try:
        env = gym.make(env_id)
    except Exception as e:
        messagebox.showerror("Environment Error", f"Error creating environment: {e}")
        return

    # Check algorithm compatibility with action space:
    # DQN requires Discrete actions; TD3/DDPG/SAC require continuous (Box).
    if algo_name == "DQN" and not isinstance(env.action_space, gym.spaces.Discrete):
        messagebox.showerror("Algorithm Error", f"DQN requires a Discrete action space for this environment.")
        env.close()
        return
    if algo_name in ["TD3", "DDPG", "SAC"] and not isinstance(env.action_space, gym.spaces.Box):
        messagebox.showerror("Algorithm Error", f"{algo_name} requires a continuous (Box) action space.")
        env.close()
        return

    match algo_name:
        case "PPO":
            model = PPO("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, verbose=1)
        case "DQN":
            model = DQN("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, buffer_size=buffer_size,
                        exploration_initial_eps=epsilon, batch_size=batch_size, verbose=1)
        case "A2C":
            model = A2C("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, verbose=1)
        case "TD3":
            model = TD3("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, buffer_size=buffer_size,
                        batch_size=batch_size, verbose=1)
        case "DDPG":
            model = DDPG("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, buffer_size=buffer_size,
                         batch_size=batch_size, verbose=1)
        case "SAC":
            model = SAC("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, buffer_size=buffer_size,
                        batch_size=batch_size, verbose=1)
        case _:
            messagebox.showerror("Algorithm Error", f"Algorithm {algo_name} is not supported.")

    print("Training started...")
    model.learn(total_timesteps=num_steps)
    print("Training completed.")
    messagebox.showinfo("Training Completed", "Training completed successfully!")
    # Ask user for a model name and save the model
    try:
        model_name = simpledialog.askstring("Save Model", "Enter a name for the trained model:")
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = None
        if model_name:
            # sanitize name
            safe_name = re.sub(r"[^0-9A-Za-z._-]", "_", model_name.strip())
            filename = f"{safe_name}_{algo_name}.zip"
            model_path = os.path.join(models_dir, filename)
            model.save(model_path)
            messagebox.showinfo("Model Saved", f"Model saved to: {model_path}")
        else:
            # if user didn't provide a name, save a temporary model to visualize
            filename = f"_tmp_visualize_{int(time.time())}_{algo_name}.zip"
            model_path = os.path.join(models_dir, filename)
            model.save(model_path)
    except Exception as e:
        messagebox.showerror("Save Error", f"Failed to save model: {e}")

    # Close the env in this process (visualization will run in a separate process)
    try:
        env.close()
    except Exception:
        pass

    # Spawn a subprocess to visualize so rendering runs in its own process
    try:
        visualize_model(env_id, model_path)
    except Exception as e:
        messagebox.showerror("Visualization Error", f"Failed to launch visualization: {e}")


start_button = ctk.CTkButton(training_frame, text="Start Training", command=start_training, font=gui_utils.DEFAULT_BUTTON_FONT)
start_button.grid(row=11, column=0, pady=12)

back_button = ctk.CTkButton(training_frame, text="Back", command=lambda: gui_utils.switch_frame(frames["HomeScreen"]), font=gui_utils.DEFAULT_BUTTON_FONT)
back_button.grid(row=12, column=0, pady=6)

# The old "Load Model" popup UI was removed; use Evaluation Mode
# for loading and visualizing saved models. The logic to list and
# open model files lives in the Evaluation screen paths.

"""
Environment editing in the UI was removed per request. To change
which display names and gym ids appear, edit the `ENV_MAP` above.
"""

# ------------------ Evaluation Screen ------------------
eval_frame = gui_utils.create_frame("Evaluation", container, frames)
eval_frame.grid_columnconfigure(0, weight=1)

eval_title = gui_utils.create_title_label(eval_frame, "Evaluation Mode")
eval_title.grid(row=0, column=0, pady=(20, 10))

# Model selection (combo populated from models/ folder)
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)
model_files = sorted([f for f in os.listdir(models_dir) if f.endswith('.zip')])
model_combo = ctk.CTkComboBox(eval_frame, values=model_files, state="readonly", font=gui_utils.DEFAULT_TEXT_FONT)
if model_files:
    model_combo.set(model_files[0])
model_combo.grid(row=1, column=0, sticky="ew", padx=12)

# Environment selection for evaluation
eval_env_combo = ctk.CTkComboBox(eval_frame, values=list(ENV_MAP.keys()), state="readonly", font=gui_utils.DEFAULT_TEXT_FONT)
eval_env_combo.set(list(ENV_MAP.keys())[0])
eval_env_combo.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 6))

# Render checkbox
render_var = ctk.BooleanVar(value=True)
render_checkbox = ctk.CTkCheckBox(eval_frame, text="Render during evaluation", variable=render_var, font=gui_utils.DEFAULT_TEXT_FONT)
render_checkbox.grid(row=3, column=0, sticky="w", padx=12)

# Status and results
eval_status = ctk.CTkLabel(eval_frame, text="Ready", font=gui_utils.DEFAULT_TEXT_FONT)
eval_status.grid(row=4, column=0, padx=12, pady=(8, 6))

results_label = ctk.CTkLabel(eval_frame, text="", wraplength=600, font=gui_utils.DEFAULT_TEXT_FONT)
results_label.grid(row=6, column=0, padx=12, pady=(6, 12))

# Control buttons
eval_buttons_frame = ctk.CTkFrame(eval_frame)
eval_buttons_frame.grid(row=5, column=0, sticky="ew", padx=12)
eval_buttons_frame.grid_columnconfigure(0, weight=1)
eval_buttons_frame.grid_columnconfigure(1, weight=1)

_eval_state = {"stop": False}

def run_evaluation():
    files = sorted([f for f in os.listdir(models_dir) if f.endswith('.zip')])
    sel = model_combo.get()
    if not sel:
        messagebox.showerror("No Model", "Please select a model to evaluate.")
        return

    model_path = os.path.join(models_dir, sel)
    env_display = eval_env_combo.get()
    env_id = ENV_MAP.get(env_display, env_display)

    try:
        env_local = gym.make(env_id)
    except Exception as e:
        messagebox.showerror("Env Error", f"Failed to create environment: {e}")
        return

    # load model (detect algorithm from filename)
    algo_detect = None
    for a in ["PPO", "DQN", "A2C", "TD3", "DDPG", "SAC"]:
        if ("_" + a + ".zip") in sel or sel.upper().endswith(f"_{a}.ZIP"):
            algo_detect = a
            break

    algo_map = {"PPO": PPO, "DQN": DQN, "A2C": A2C, "TD3": TD3, "DDPG": DDPG, "SAC": SAC}
    model_loaded = None
    try:
        if algo_detect and algo_detect in algo_map:
            ModelClass = algo_map[algo_detect]
            model_loaded = ModelClass.load(model_path, env=env_local)
        else:
            for ModelClass in algo_map.values():
                try:
                    model_loaded = ModelClass.load(model_path, env=env_local)
                    break
                except Exception:
                    continue
    except Exception as e:
        messagebox.showerror("Load Error", f"Failed to load model: {e}")
        try:
            env_local.close()
        except Exception:
            pass
        return

    if model_loaded is None:
        messagebox.showerror("Load Error", "Could not determine model type or load the model.")
        try:
            env_local.close()
        except Exception:
            pass
        return

    # Evaluation parameters
    max_steps = 5000

    # If rendering is requested, delegate evaluation to the worker process
    # which will perform rendering on its own main thread (avoids macOS crashes).
    if render_var.get():
        # create temp JSON file path for worker to write stats
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix="_eval_stats.json")
        tmpf.close()
        worker_py = os.path.join(os.getcwd(), "src", "visualize_worker.py")
        # heartbeat file for progress updates
        hb = tempfile.NamedTemporaryFile(delete=False, suffix="_eval_heartbeat.json")
        hb.close()
        args = [sys.executable, worker_py, "--model", model_path, "--env", env_id,
            "--eval-steps", str(max_steps), "--stats-out", tmpf.name, "--heartbeat", hb.name]
        # include render flag
        args += ["--render"]

        try:
            proc = subprocess.Popen(args)
        except Exception as e:
            messagebox.showerror("Evaluation Error", f"Failed to start evaluation worker: {e}")
            try:
                env_local.close()
            except Exception:
                pass
            try:
                os.unlink(hb.name)
            except Exception:
                pass
            return

        eval_status.configure(text=f"Running: 0 / {max_steps}")
        results_label.configure(text="Waiting for worker to finish...")

        # Provide a Stop button that terminates the worker
        def stop_worker():
            try:
                proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            eval_status.configure(text="Stopped by user")

        stop_btn.configure(command=stop_worker)

        # Polling loop to check for worker completion and read heartbeat/stats JSON
        def poll_worker():
            # update from heartbeat if available
            try:
                if os.path.exists(hb.name):
                    with open(hb.name, 'r') as fh:
                        data_h = json.load(fh)
                    step_h = int(data_h.get('step', 0))
                    try:
                        eval_status.configure(text=f"Running: {step_h} / {max_steps}")
                    except Exception:
                        pass
                    if data_h.get('done'):
                        # worker finished; fallthrough to reading stats
                        pass
            except Exception:
                pass

            ret = proc.poll()
            if ret is None:
                root.after(200, poll_worker)
                return

            # process exited; try to read stats
            try:
                with open(tmpf.name, 'r') as fh:
                    data = json.load(fh)

                # worker writes keys like mean_reward, std_reward, etc.
                episodes = data.get('episodes', 0)
                mean_r = data.get('mean_reward', 0.0)
                std_r = data.get('std_reward', 0.0)
                min_r = data.get('min_reward', 0.0)
                max_r = data.get('max_reward', 0.0)
                median_r = data.get('median_reward', 0.0)
                avg_len = data.get('avg_episode_length', 0.0)

                txt = (
                    f"Episodes: {episodes}\n"
                    f"Mean reward: {mean_r:.3f}\n"
                    f"Std reward: {std_r:.3f}\n"
                    f"Min reward: {min_r:.3f}\n"
                    f"Max reward: {max_r:.3f}\n"
                    f"Median reward: {median_r:.3f}\n"
                    f"Avg episode length: {avg_len:.1f}\n"
                )
                results_label.configure(text=txt)
                eval_status.configure(text="Evaluation completed")
            except Exception as e:
                results_label.configure(text=f"Worker finished but failed to read stats: {e}")
                eval_status.configure(text="Worker finished")

            # cleanup
            try:
                os.unlink(tmpf.name)
            except Exception:
                pass
            try:
                env_local.close()
            except Exception:
                pass

        # start polling
        root.after(500, poll_worker)
        return

    # Otherwise perform headless (non-render) evaluation in-process
    # Use shared eval state so Stop button can signal it
    _eval_state.update({
        "step": 0,
        "stop": False,
        "episode_reward": 0.0,
        "episode_len": 0,
        "episode_rewards": [],
        "episode_lengths": []
    })
    state = _eval_state

    reset_out = env_local.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    eval_status.configure(text=f"Running 0 / {max_steps}")
    results_label.configure(text="")

    def eval_step():
        nonlocal obs
        if state["stop"]:
            eval_status.configure(text="Stopped by user")
            try:
                env_local.close()
            except Exception:
                pass
            return

        if state["step"] >= max_steps:
            # finalize
            if state["episode_len"] > 0:
                state["episode_rewards"].append(state["episode_reward"])
                state["episode_lengths"].append(state["episode_len"])

            # compute stats
            if state["episode_rewards"]:
                arr = np.array(state["episode_rewards"])
                mean_r = float(np.mean(arr))
                std_r = float(np.std(arr))
                min_r = float(np.min(arr))
                max_r = float(np.max(arr))
                median_r = float(np.median(arr))
                count = len(arr)
                avg_len = float(np.mean(state["episode_lengths"])) if state["episode_lengths"] else 0.0
                txt = (
                    f"Episodes: {count}\n"
                    f"Mean reward: {mean_r:.3f}\n"
                    f"Std reward: {std_r:.3f}\n"
                    f"Min reward: {min_r:.3f}\n"
                    f"Max reward: {max_r:.3f}\n"
                    f"Median reward: {median_r:.3f}\n"
                    f"Avg episode length: {avg_len:.1f}\n"
                )
            else:
                txt = "No complete episodes during evaluation."

            results_label.configure(text=txt)
            eval_status.configure(text="Evaluation completed")
            try:
                env_local.close()
            except Exception:
                pass
            return

        # step
        try:
            action, _ = model_loaded.predict(obs, deterministic=True)
            step_out = env_local.step(action)
        except Exception as e:
            messagebox.showerror("Runtime Error", f"Error during evaluation step: {e}")
            try:
                env_local.close()
            except Exception:
                pass
            return

        if len(step_out) == 4:
            obs, reward, done, info = step_out
        else:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated

        state["episode_reward"] += float(reward)
        state["episode_len"] += 1

        # no rendering in this branch

        if done:
            state["episode_rewards"].append(state["episode_reward"])
            state["episode_lengths"].append(state["episode_len"])
            state["episode_reward"] = 0.0
            state["episode_len"] = 0
            # reset
            reset_out = env_local.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        state["step"] += 1
        if state["step"] % 10 == 0:
            eval_status.configure(text=f"Running {state['step']} / {max_steps}")

        root.after(1, eval_step)

    # start
    root.after(1, eval_step)

def stop_evaluation():
    _eval_state["stop"] = True

run_btn = ctk.CTkButton(eval_buttons_frame, text="Run Evaluation", command=run_evaluation, font=gui_utils.DEFAULT_BUTTON_FONT)
run_btn.grid(row=0, column=0, sticky="ew", padx=(0,6))
stop_btn = ctk.CTkButton(eval_buttons_frame, text="Stop", fg_color="#D9534F", command=stop_evaluation, font=gui_utils.DEFAULT_BUTTON_FONT)
stop_btn.grid(row=0, column=1, sticky="ew", padx=(6,0))

# Back button 
back_eval_btn = ctk.CTkButton(eval_frame, text="Back", command=lambda: gui_utils.switch_frame(frames["HomeScreen"]), font=gui_utils.DEFAULT_BUTTON_FONT)
back_eval_btn.grid(row=7, column=0, pady=10)

# settings screen
settings_frame = gui_utils.create_frame("Settings", container, frames)
settings_frame.grid_columnconfigure(0, weight=1)

settings_title = gui_utils.create_title_label(settings_frame, "Settings")
settings_title.grid(row=0, column=0, pady=(20, 8))

# dark/light modes
appearance_label = gui_utils.create_text_label(settings_frame, "Appearance:")
appearance_label.grid(row=1, column=0, sticky="w", padx=12)
appearance_combo = ctk.CTkComboBox(settings_frame, values=["System", "Light", "Dark"], state="readonly", font=gui_utils.DEFAULT_TEXT_FONT)
appearance_combo.set("System")
appearance_combo.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))

# selection
mode_label = gui_utils.create_text_label(settings_frame, "Mode:")
mode_label.grid(row=3, column=0, sticky="w", padx=12)
mode_combo = ctk.CTkComboBox(settings_frame, values=["User", "Power User", "Dev"], state="readonly", font=gui_utils.DEFAULT_TEXT_FONT)
mode_combo.set("Power User")
mode_combo.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 8))

# explanation
settings_help = gui_utils.create_text_label(settings_frame, "Modes:\n- User: minimal UI (no Evaluation).\n- Power User: most features.\n- Dev: full features + debug logs.")
settings_help.grid(row=5, column=0, padx=12, pady=(6,12))

def apply_mode(mode: str):
    """Apply the selected mode to the UI by showing/hiding/fading controls."""
    # store current mode
    global SETTINGS
    SETTINGS = SETTINGS if 'SETTINGS' in globals() else {}
    SETTINGS['mode'] = mode

    # training controls: which slider frames to hide for lighter modes
    # Always keep timesteps_slider visible for all modes
    advanced_frames = [lr_slider, gamma_slider, batch_size_slider, buffer_size_slider, epsilon_slider]

    if mode == 'User':
        # hide all advanced controls except timesteps
        for f in advanced_frames:
            try:
                f.grid_remove()
            except Exception:
                pass
        # disable evaluation access
        try:
            eval_button.configure(state='disabled')
        except Exception:
            pass

    elif mode == 'Power User':
        # show common hyperparams, hide niche ones
        for f in [lr_slider, gamma_slider, batch_size_slider]:
            try:
                f.grid()
            except Exception:
                pass
        for f in [buffer_size_slider, epsilon_slider]:
            try:
                f.grid_remove()
            except Exception:
                pass
        try:
            eval_button.configure(state='normal')
        except Exception:
            pass

    else:  # Dev
        for f in advanced_frames:
            try:
                f.grid()
            except Exception:
                pass
        try:
            eval_button.configure(state='normal')
        except Exception:
            pass

def apply_appearance(value: str):
    try:
        if value == 'System':
            ctk.set_appearance_mode('system')
        elif value == 'Light':
            ctk.set_appearance_mode('light')
        else:
            ctk.set_appearance_mode('dark')
    except Exception:
        pass

def on_settings_change(_=None):
    apply_appearance(appearance_combo.get())
    apply_mode(mode_combo.get())

apply_btn = ctk.CTkButton(settings_frame, text="Apply", command=on_settings_change, font=gui_utils.DEFAULT_BUTTON_FONT)
apply_btn.grid(row=6, column=0, pady=10)

# Back button for Settings
settings_back = ctk.CTkButton(settings_frame, text="Back", command=lambda: gui_utils.switch_frame(frames["HomeScreen"]), font=gui_utils.DEFAULT_BUTTON_FONT)
settings_back.grid(row=7, column=0, pady=6)

# initialize settings
SETTINGS = {'mode': mode_combo.get(), 'appearance': appearance_combo.get()}
on_settings_change()


# About Page
about_frame = gui_utils.create_frame("AboutPage", container, frames)

title_label = gui_utils.create_title_label(about_frame, text="About")
about_text = gui_utils.create_text_label(about_frame, "DQNSuite is an experimental GUI for training and evaluating reinforcement learning agents using Stable-Baselines3. This release focuses on usability and evaluation workflows.\n\nFeatures:\n- Train agents using PPO, DQN, A2C, TD3, DDPG, SAC\n- Save and load models to ./models\n- Evaluation mode with stoppable runs and reward statistics\n- Simple UI with responsive layout and configurable environments (edit `ENV_MAP` in code)", 600)
about_text2 = gui_utils.create_text_label(about_frame, f"Version: {VERSION} — Developed by DQN Labs, updated Dec 2025.")

about_frame.grid_columnconfigure(0, weight=1)
title_label.grid(row=0, column=0, pady=(20, 8))
about_text.grid(row=1, column=0, padx=12)
about_text2.grid(row=2, column=0, padx=12, pady=(6, 12))

back_button = ctk.CTkButton(about_frame, text="Back", command=lambda: gui_utils.switch_frame(frames["HomeScreen"]), font=gui_utils.DEFAULT_BUTTON_FONT)
back_button.grid(row=3, column=0, pady=10)

# Documentation Page
doc_frame = gui_utils.create_frame("Documentation", container, frames)
title_label = gui_utils.create_title_label(doc_frame, text="Documentation")

doc_intro = gui_utils.create_text_label(doc_frame, "Welcome to the DQNSuite documentation. Use this page to learn how to train, evaluate, and manage models. The text area below contains full usage instructions and tips.", 600)

doc_frame.grid_columnconfigure(0, weight=1)
title_label.grid(row=0, column=0, pady=(20, 8))
doc_intro.grid(row=1, column=0, padx=12)

back_button = ctk.CTkButton(doc_frame, text="Back", command=lambda: gui_utils.switch_frame(frames["HomeScreen"]), font=gui_utils.DEFAULT_BUTTON_FONT)
back_button.grid(row=3, column=0, pady=10)

# Read docs file and present in a scrollable textbox
docs: str
with open('data/docs_page_1.txt', 'r') as file:
    docs = file.read()

extra = "\n\n=== Quick Start ===\n1) Pick an environment from the Training Hub (edit `ENV_MAP` in code to add custom names).\n2) Choose an algorithm and hyperparameters, then click Start Training.\n3) After training completes, save the model when prompted (saved to `./models`).\n4) Switch to Evaluation Mode to load a model and run a 5,000-step evaluation (stoppable).\n\n=== Tips ===\n- For continuous control use TD3/DDPG/SAC; for discrete actions use DQN/PPO/A2C.\n- If a gym environment depends on extra packages (e.g., Box2D), install them separately.\n- To change fonts, edit `src/gui_utils.py` DEFAULT_*_FONT values.\n\n=== Troubleshooting ===\n- If rendering causes crashes on Mac, try running without rendering or use headless display drivers.\n- If a model fails to load, ensure it was saved with a supported algorithm suffix in the filename (e.g. `_PPO.zip`).\n"

doc_box = ctk.CTkTextbox(doc_frame, width=860, height=420, font=gui_utils.DEFAULT_TEXT_FONT)
doc_box.insert("0.0", docs + extra)
doc_box.configure(state="disabled")
doc_box.grid(row=4, column=0, padx=12, pady=(12, 20))


# Visualize the trained model in a separate process to avoid native
# rendering/GIL crashes on macOS. This spawns `src/visualize_worker.py`
# in a child process which performs rendering on its main thread.
def visualize_model(env_id: str, model_path: str, max_steps: int = 1_000_000) -> None:
    worker_py = os.path.join(os.getcwd(), "src", "visualize_worker.py")
    # heartbeat file for progress updates
    hb = tempfile.NamedTemporaryFile(delete=False, suffix="_viz_heartbeat.json")
    hb.close()

    args = [sys.executable, worker_py, "--model", model_path, "--env", env_id, "--max-steps", str(max_steps), "--heartbeat", hb.name, "--render"]

    try:
        proc = subprocess.Popen(args)
    except Exception as e:
        raise RuntimeError(f"Failed to launch visualization worker: {e}")

    # Create a small control window to allow the user to stop the worker
    control_win = ctk.CTkToplevel(root)
    control_win.title("Visualization")
    control_win.geometry("300x120")
    control_win.transient(root)

    status_label = ctk.CTkLabel(control_win, text=f"Running: 0 / {max_steps}")
    status_label.pack(pady=(12, 6))

    def stop_proc():
        try:
            proc.terminate()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    stop_btn = ctk.CTkButton(control_win, text="Stop", fg_color="#D9534F", command=stop_proc)
    stop_btn.pack(pady=(0, 12))

    # Poll heartbeat file for progress and watch process exit
    def poll():
        # update from heartbeat if available
        try:
            if os.path.exists(hb.name):
                with open(hb.name, 'r') as fh:
                    data = json.load(fh)
                step = int(data.get('step', 0))
                try:
                    status_label.configure(text=f"Running: {step} / {max_steps}")
                except Exception:
                    pass
                if data.get('done'):
                    # worker reported done; close UI
                    try:
                        control_win.destroy()
                    except Exception:
                        pass
                    try:
                        os.unlink(hb.name)
                    except Exception:
                        pass
                    return
        except Exception:
            pass

        # if process ended, close UI
        ret = proc.poll()
        if ret is not None:
            try:
                control_win.destroy()
            except Exception:
                pass
            try:
                os.unlink(hb.name)
            except Exception:
                pass
            return

        root.after(200, poll)

    root.after(200, poll)


# Start the application on the Home Screen
gui_utils.switch_frame(home_frame)

root.mainloop()
