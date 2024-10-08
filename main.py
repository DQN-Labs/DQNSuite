from tkinter import messagebox
import customtkinter as ctk
import gym
from stable_baselines3 import PPO, DQN, A2C, TD3, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

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

def switch_frame(frame):
    frame.tkraise()

# Home Screen
home_frame = ctk.CTkFrame(container)
frames["HomeScreen"] = home_frame
home_frame.grid(row=0, column=0, sticky="nsew")

title_label = ctk.CTkLabel(home_frame, text="DQNSuite", font=("Helvetica", 24))
title_label.pack(pady=(50, 20))

training_button = ctk.CTkButton(home_frame, text="Training Hub", command=lambda: switch_frame(frames["TrainingHub"]))
training_button.pack(pady=10)

about_button = ctk.CTkButton(home_frame, text="About", command=lambda: switch_frame(frames["AboutPage"]))
about_button.pack(pady=10)

doc_button = ctk.CTkButton(home_frame, text="Documentation", command=lambda: switch_frame(frames["Documentation"]))
doc_button.pack(pady=10)

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
training_frame = ctk.CTkFrame(container)
frames["TrainingHub"] = training_frame
training_frame.grid(row=0, column=0, sticky="nsew")

title_label = ctk.CTkLabel(training_frame, text="Training Hub", font=("Helvetica", 24))
title_label.pack(pady=(50, 20))

env_label = ctk.CTkLabel(training_frame, text="Select Environment:")
env_label.pack(pady=5)
env_var = ctk.CTkComboBox(training_frame, values=["CartPole-v1", "MountainCar-v0", "LunarLander-v2", "Acrobot-v1", "Pendulum-v0"], state="readonly")
env_var.set("CartPole-v1")
env_var.pack(pady=5)

algo_label = ctk.CTkLabel(training_frame, text="Select Algorithm:")
algo_label.pack(pady=5)
algo_var = ctk.CTkComboBox(training_frame, values=["PPO", "DQN", "A2C", "TD3", "DDPG", "SAC"], state="readonly")
algo_var.set("PPO")
algo_var.pack(pady=5)

# Hyperparameter sliders and labels
def create_hyperparameter_slider(parent, text, from_, to, resolution, initial_value):
    frame = ctk.CTkFrame(parent)
    frame.pack(pady=5, fill="x")

    label = ctk.CTkLabel(frame, text=text)
    label.pack(side="left", padx=10)

    value_label = ctk.CTkLabel(frame, text=f"{initial_value:.5f}")
    value_label.pack(side="right", padx=10)

    slider = ctk.CTkSlider(frame, from_=from_, to=to, number_of_steps=(to-from_)/resolution, width=400)
    slider.set(initial_value)
    slider.pack(side="left", padx=10, fill="x", expand=True)

    def update_value_label(event):
        value_label.configure(text=f"{slider.get():.5f}")

    slider.bind("<Motion>", update_value_label)

    return slider

# Add sliders for hyperparameters
lr_slider = create_hyperparameter_slider(training_frame, "Learning Rate:", 0.0001, 0.01, 0.001, 0.001)
gamma_slider = create_hyperparameter_slider(training_frame, "Gamma:", 0.9, 0.9999, 0.001, 0.99)
batch_size_slider = create_hyperparameter_slider(training_frame, "Batch Size:", 8, 256, 1, 64)
buffer_size_slider = create_hyperparameter_slider(training_frame, "Buffer Size:", 1000, 100000, 1, 10000)
epsilon_slider = create_hyperparameter_slider(training_frame, "Epsilon (DQN):", 0.01, 1.0, 0.01, 0.1)
timesteps_slider = create_hyperparameter_slider(training_frame, "Number of Timesteps:", 1000, 1000000, 1, 10000)

def start_training():
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

    if algo_name == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, verbose=1)
    elif algo_name == "DQN":
        model = DQN("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, buffer_size=buffer_size, exploration_initial_eps=epsilon, batch_size=batch_size, verbose=1)
    elif algo_name == "A2C":
        model = A2C("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, verbose=1)
    elif algo_name == "TD3":
        model = TD3("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, buffer_size=buffer_size, batch_size=batch_size, verbose=1)
    elif algo_name == "DDPG":
        model = DDPG("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, buffer_size=buffer_size, batch_size=batch_size, verbose=1)
    elif algo_name == "SAC":
        model = SAC("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, buffer_size=buffer_size, batch_size=batch_size, verbose=1)

    print("Training started...")
    model.learn(total_timesteps=num_steps)
    print("Training completed.")
    messagebox.showinfo("Training Completed", "Training completed successfully!")
    visualize_model(env, model)
    env.close()

start_button = ctk.CTkButton(training_frame, text="Start Training", command=start_training)
start_button.pack(pady=10)

back_button = ctk.CTkButton(training_frame, text="Back", command=lambda: switch_frame(frames["HomeScreen"]))
back_button.pack(pady=10)

# About Page
about_frame = ctk.CTkFrame(container)
frames["AboutPage"] = about_frame
about_frame.grid(row=0, column=0, sticky="nsew")

title_label = ctk.CTkLabel(about_frame, text="About", font=("Helvetica", 24))
title_label.pack(pady=(50, 20))

about_text = ctk.CTkLabel(about_frame, text="This application allows users to train reinforcement learning agents using popular algorithms and environments.", wraplength=600)
about_text.pack(pady=10)

dev_info_label = ctk.CTkLabel(about_frame, text="Developed by DQN Labs, May 27, 2024.")
dev_info_label.pack(pady=10)

back_button = ctk.CTkButton(about_frame, text="Back", command=lambda: switch_frame(frames["HomeScreen"]))
back_button.pack(pady=10)

# Documentation Page
doc_frame = ctk.CTkFrame(container)
frames["Documentation"] = doc_frame
doc_frame.grid(row=0, column=0, sticky="nsew")

title_label = ctk.CTkLabel(doc_frame, text="Documentation", font=("Helvetica", 24))
title_label.pack(pady=(50, 20))

doc_text = ctk.CTkLabel(doc_frame, text="Welcome to the DQNSuite! Before you embark on your journey, here's a quick overview on how to use the application.", wraplength=600)
doc_text.pack(pady=10)

back_button = ctk.CTkButton(doc_frame, text="Back", command=lambda: switch_frame(frames["HomeScreen"]))
back_button.pack(pady=40)

main_doc_text = ctk.CTkLabel(doc_frame, text='''The Training Hub is your main way of accessing the suite. You first select an environment
from the ones listed. Once you have found a suitable environment, you then seleect your algorithm
for reinforcement learning. PPO, TD3, DDPG, and SAC are usually the best for continuous spaces. A2C is a versatile
algorithm suitable for almost any environment. DQN is a great algorithm for discrete spaces, in fact
it does not support continuous environemnts. You can then adjust the hyperparameters as such.
The learning rate increases the speed at which the AI learns, but can also make it develop bad 
habits which it cannot lose later on. Gamma is the discount factor for the AI's reward system. 
A higher gamma gives lower positive rewards but lower negative rewards and vice versa. Batch size
and buffer size are always better to have more of, but it can heavily impact the performance
of your computer. Only use high settings for these if you have a high-end PC. Epsilon 
is the exploration rate of the AI. It is only applicable when used with DQN> The number of timesteps
signifies how many episodes the AI will train. More timesteps will take longer, but will ofter give 
better results. Once you are done adjusting, you can finally click the Start Training button. Training 
may take a few moments, so please keep patience. Once training is done, a notification will pop up, 
informing you of the end of the session. Once you dismiss the notification, the AI will be tested live
in front of you for a brief period, so you can see the result of the training. ''')
main_doc_text.pack(pady=(40, 20))

# Function to visualize the trained model
def visualize_model(env, model):
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()

# Start the application on the Home Screen
switch_frame(home_frame)

# Run the application
root.mainloop()
