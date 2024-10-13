from tkinter import messagebox
import customtkinter as ctk
import gym
from stable_baselines3 import PPO, DQN, A2C, TD3, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import gui_utils

# Initialize CustomTkinter
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

# Create the main application window
root = ctk.CTk()
root.title("DQNSuite")
root.geometry("675x574")

# Create a container frame
container = ctk.CTkFrame(root)
container.pack(fill="both", expand=True)

# Define frames as a dictionary
frames = {}


# Home Screen
home_frame = gui_utils.create_frame("HomeScreen", container, frames)
title_label = gui_utils.create_title_label(home_frame, "DQNSuite")

training_button = gui_utils.create_nav_button(home_frame, "Training Hub", frames, "TrainingHub")
about_button = gui_utils.create_nav_button(home_frame, "About", frames, "AboutPage")
doc_button = gui_utils.create_nav_button(home_frame, "Documentation", frames, "Documentation")

def center_widgets(event):
    home_frame.update_idletasks()
    screen_width = home_frame.winfo_width()
    screen_height = home_frame.winfo_height()

    title_label.place(x=(screen_width - title_label.winfo_width()) / 2, y=50)
    training_button.place(x=(screen_width - training_button.winfo_width()) / 2, y=150)
    about_button.place(x=(screen_width - about_button.winfo_width()) / 2, y=200)
    doc_button.place(x=(screen_width - doc_button.winfo_width()) / 2, y=250)

home_frame.bind("<Configure>", center_widgets)

# Training Hub
training_frame = gui_utils.create_frame("TrainingHub", container, frames)
title_label = gui_utils.create_title_label(training_frame, "Training Parameters")
env_label = gui_utils.create_text_label(training_frame, "Select Environment:", pady=5)
env_var = ctk.CTkComboBox(training_frame, values=[
    "CartPole-v1", "MountainCar-v0", "LunarLander-v2", "Acrobot-v1", "Pendulum-v1"
], state="readonly")
env_var.set("CartPole-v1")
env_var.pack(pady=5)

algo_label = gui_utils.create_text_label(training_frame, "Select Algorithm:", pady=5)

algo_var = ctk.CTkComboBox(training_frame, values=["PPO", "DQN", "A2C", "TD3", "DDPG", "SAC"], state="readonly")
algo_var.set("PPO")
algo_var.pack(pady=5)

# Add sliders for hyperparameters
lr_slider = gui_utils.create_hyperparameter_slider(training_frame, "Learning Rate:", 0.0001, 0.01, 0.001, 0.001)
gamma_slider = gui_utils.create_hyperparameter_slider(training_frame, "Gamma:", 0.9, 0.9999, 0.001, 0.99)
batch_size_slider = gui_utils.create_hyperparameter_slider(training_frame, "Batch Size:", 8, 256, 1, 64)
buffer_size_slider = gui_utils.create_hyperparameter_slider(training_frame, "Buffer Size:", 1000, 100000, 1, 10000)
epsilon_slider = gui_utils.create_hyperparameter_slider(training_frame, "Epsilon (DQN):", 0.01, 1.0, 0.01, 0.1)
timesteps_slider = gui_utils.create_hyperparameter_slider(training_frame, "Number of Timesteps:", 1000, 1000000, 1, 10000)


def start_training() -> None:
    env_name = env_var.get()
    algo_name = algo_var.get()
    learning_rate = lr_slider.get()
    gamma = gamma_slider.get()
    batch_size = int(batch_size_slider.get())
    buffer_size = int(buffer_size_slider.get())
    epsilon = epsilon_slider.get()
    num_steps = int(timesteps_slider.get())

    try:
        env = gym.make(env_name)
    except gym.error.Error as e:
        messagebox.showerror("Environment Error", f"Error creating environment: {e}")
        return

    # Check for algorithm compatibility with action space
    if algo_name in ["DQN", "TD3", "DDPG", "SAC"] and isinstance(env.action_space, gym.spaces.Box) is False:
        messagebox.showerror("Algorithm Error", f"{algo_name} is not supported by this environment.")
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
    visualize_model(env, model)
    env.close()


start_button = ctk.CTkButton(training_frame, text="Start Training", command=start_training)
start_button.pack(pady=10)

back_button = ctk.CTkButton(training_frame, text="Back", command=lambda: gui_utils.switch_frame(frames["HomeScreen"]))
back_button.pack(pady=10)

# About Page
about_frame = gui_utils.create_frame("AboutPage", container, frames)

title_label = gui_utils.create_title_label(about_frame, text="About")
gui_utils.create_text_label(about_frame, "This application allows users to train reinforcement learning agents using popular algorithms and environments.", 600)
gui_utils.create_text_label(about_frame, "Developed by DQN Labs, May 27, 2024.")

back_button = ctk.CTkButton(about_frame, text="Back", command=lambda: gui_utils.switch_frame(frames["HomeScreen"]))
back_button.pack(pady=10)

# Documentation Page
doc_frame = gui_utils.create_frame("Documentation", container, frames)
title_label = gui_utils.create_title_label(doc_frame, text="Documentation")

gui_utils.create_text_label(doc_frame, "Welcome to the DQNSuite! Before you embark on your journey, here's a quick overview on how to use the application.", 600)

back_button = ctk.CTkButton(doc_frame, text="Back", command=lambda: gui_utils.switch_frame(frames["HomeScreen"]))
back_button.pack(pady=40)

docs: str
with open('../data/docs_page_1.txt', 'r') as file:
    docs = file.read()

main_doc_text = ctk.CTkLabel(doc_frame, text=docs)
main_doc_text.pack(pady=(40, 20))


# Function to visualize the trained model
def visualize_model(env, model) -> None:
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()


# Start the application on the Home Screen
gui_utils.switch_frame(home_frame)

root.mainloop()
