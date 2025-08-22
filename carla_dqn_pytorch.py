import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('Carla/DQN')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from threading import Thread

from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 4
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Resnet18"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 15

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10



class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3].reshape(3,self.im_height, self.im_width)
        #print(i3.(3,self.im_height, self.im_widreshapeth))
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.optimizer = self.get_optim()
        self.target_model = self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        #self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        #self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False
        #self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        #base_model = models.resnet18(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH,3))
        base_model = models.resnet18(pretrained=True)

        last_layer_in = base_model.fc.in_features

        base_model.fc = nn.Linear(last_layer_in, 3)
        #model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])

        return base_model.to(device)

    def get_optim(self):
        #base_model = models.resnet18(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH,3))
        optimizer = optim.Adam( self.model.parameters(), lr = 0.001)

        return optimizer

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])/255
        #with self.graph.as_default():
        self.model.eval()
        current_qs_list = self.model(torch.from_numpy(current_states).to(device,dtype=torch.float32))#, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        #with self.graph.as_default():
        self.model.eval()
        future_qs_list = self.target_model(torch.from_numpy(new_current_states).to(device,dtype=torch.float32)).cpu().detach().numpy()#, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                #max_future_q = torch.max(future_qs_list[index]).cpu().detach().numpy()
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        #if self.tensorboard.step > self.last_logged_episode:
         #   log_this_step = True
         #   self.last_log_episode = self.tensorboard.step

        #with self.graph.as_default():
        self.model.train()
        #target = torch.Tensor(y).to(device,dtype=torch.float32)
        target_np = np.array( [elem for singleList in y for elem in singleList],dtype=float).reshape(MINIBATCH_SIZE,3)
        target = torch.from_numpy(target_np).to(device,dtype=torch.float32)
        output = self.model(torch.from_numpy(np.array(X)/255).to(device,dtype=torch.float32))#,batch_size=TRAINING_BATCH_SIZE)
        loss = F.smooth_l1_loss(output, target)
        writer.add_scalar('Trainloss',loss.item())
        loss.backward()
        self.optimizer.step()

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        self.model.eval()
        return self.model(torch.from_numpy(np.array(state).reshape(-1, *state.shape)/255).to(device,dtype=torch.float))[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1,3, IM_HEIGHT, IM_WIDTH)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        #with self.graph.as_default():
        self.model.train()
        output = self.model(torch.from_numpy(X).to(device,dtype=torch.float32))
        loss = F.smooth_l1_loss(output, torch.from_numpy(y).to(device))
        writer.add_scalar('Trainloss_inloop',loss.item())
        loss.backward()
        self.optimizer.step()

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



if __name__ == '__main__':
    FPS = 60
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    #tf.random.set_seed(1)

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()


    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((3,env.im_height, env.im_width)))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:

            env.collision_hist = []

            # Update tensorboard step every episode
            writer.add_scalar('Episode',episode) 

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1

            # Reset environment and get initial state
            current_state = env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            episode_start = time.time()

            # Play for given number of seconds only
            while True:

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state).cpu().detach().numpy())
                else:
                    # Get random action
                    action = np.random.randint(0, 3)
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)

                new_state, reward, done, _ = env.step(action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1

                if done:
                    break

            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()

            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                #agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=max_reward, reward_max=max_reward, epsilon=epsilon)
                writer.add_scalar('reward_avg',average_reward,episode)
                writer.add_scalar('reward_min',max_reward,episode)
                writer.add_scalar('reward_max',max_reward,episode)
                writer.add_scalar('epsilon',epsilon,episode)

                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    torch.save(agent.model.state_dict(),(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.pth'))

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    torch.save(agent.model.state_dict(),f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.pth')
