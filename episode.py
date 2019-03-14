""" Contains the Episodes for Navigation. """
import random
import torch
import time
import sys
from constants import GOAL_SUCCESS_REWARD, STEP_PENALTY, BASIC_ACTIONS, PROCESS_REWARD, FAILED_ACTION_PENALTY
from environment import Environment
from utils.net_util import gpuify


class Episode:
    """ Episode for Navigation. """
    def __init__(self, args, gpu_id, rank, strict_done=False):
        super(Episode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None

        self.seed = args.seed + rank
        random.seed(self.seed)

        with open('./datasets/objects/int_objects.txt') as f:
            int_objects = [s.strip() for s in f.readlines()]
        with open('./datasets/objects/rec_objects.txt') as f:
            rec_objects = [s.strip() for s in f.readlines()]
        
        self.objects = int_objects + rec_objects

        self.actions_list = [{'action':a} for a in BASIC_ACTIONS]
        self.actions_taken = []

        self.locate_tomato = 0
        self.open_mic = 0
        self.place_tomato = 0
        self.close_mic = 0

    @property
    def environment(self):
        return self._env

    def state_for_agent(self):
        return self.environment.current_frame

    def step(self, action_as_int):
        action = self.actions_list[action_as_int]
        self.actions_taken.append(action)
        return self.action_step(action)

    def action_step(self, action):
        self.environment.step(action)
        reward, terminal, action_was_successful = self.judge(action)

        return reward, terminal, action_was_successful

    def slow_replay(self, delay=0.2):
        # Reset the episode
        self._env.reset(self.cur_scene, change_seed = False)
        
        for action in self.actions_taken:
            self.action_step(action)
            time.sleep(delay)
    
    def judge(self, action):
        """ Judge the last event. """
        # immediate reward
        reward = STEP_PENALTY 
        done = False
        action_was_successful = self.environment.last_action_success

        # if action['action'] == 'Done':
        #     done = True
        #     objects = self._env.last_event.metadata['objects']
        #     visible_objects = [o['objectType'] for o in objects if o['visible']]
        #     if self.target in visible_objects:
        #         reward += GOAL_SUCCESS_REWARD
        #         self.success = True

        if action['action'] == 'PickupObject':
            self.locate_tomato += 1
            objects = self._env.last_event.metadata['objects']
            visible_objects = [o['objectType'] for o in objects if o['visible']]
            if self.target[0] in visible_objects:
                self.has_tomato = True
                self.tomato_success = True

                tomato_id = [o['objectId'] for o in objects if o['objectType'] == self.target[0]][0]
                event =self._env.step(dict(action='PickupObject', objectId=tomato_id))
                self.last_event.metadata['lastActionSuccess'] = event.metadata['lastActionSuccess']

                if self.locate_tomato == 1:
                    reward += PROCESS_REWARD
            else:
                reward += FAILED_ACTION_PENALTY

        if action['action'] == 'OpenObject':
            self.open_mic += 1
            objects = self._env.last_event.metadata['objects']
            visible_objects = [o['objectType'] for o in objects if o['visible']]
            if self.target[1] in visible_objects:
                self.mic = True
                self.open_success = True

                microwave_id = [o['objectId'] for o in objects if o['objectType'] == self.target[1]][0]
                event =self._env.step(dict(action='OpenObject', objectId=microwave_id))
                self.last_event.metadata['lastActionSuccess'] = event.metadata['lastActionSuccess']

                if self.open_mic == 1:
                    reward += PROCESS_REWARD
            else:
                reward += FAILED_ACTION_PENALTY

        if action['action'] == 'PlaceHeldObject':
            self.place_tomato += 1
            objects = self._env.last_event.metadata['objects']
            visible_objects = [o['objectType'] for o in objects if o['visible']]
            if self.target[1] in visible_objects and self.has_tomato and self.mic:
                self.has_tomato = False
                self.tomato_in_mic = True
                self.put_success = True

                microwave_id = [o['objectId'] for o in objects if o['objectType'] == self.target[1]][0]
                event = self._env.step(dict(action='PlaceHeldObject', objectId=microwave_id))
                self.last_event.metadata['lastActionSuccess'] = event.metadata['lastActionSuccess']

                if self.place_tomato == 1:
                    reward += PROCESS_REWARD
            else:
                reward += FAILED_ACTION_PENALTY

        if action['action'] == 'CloseObject':
            self.close_mic += 1
            objects = self._env.last_event.metadata['objects']
            visible_objects = [o['objectType'] for o in objects if o['visible']]
            if self.target[1] in visible_objects and self.mic and self.tomato_in_mic:
                self.mic = False
                self.close_success = True

                microwave_id = [o['objectId'] for o in objects if o['objectType'] == self.target[1]][0]
                event =self._env.step(dict(action='CloseObject', objectId=microwave_id))
                self.last_event.metadata['lastActionSuccess'] = event.metadata['lastActionSuccess']

                if self.close_mic == 1:
                    reward += PROCESS_REWARD
            elif self.target[1] in visible_objects and self.mic:
                self.mic = False
                reward += FAILED_ACTION_PENALTY
            else:
                reward += FAILED_ACTION_PENALTY


        if self.open_success and self.put_success and self.close_success and self.tomato_success:
            self.success = True
            # reward += GOAL_SUCCESS_REWARD

        if self.locate_tomato > 0 and self.open_mic > 0 and self.place_tomato > 0 and self.close_mic > 0:
            done = True


        return reward, done, action_was_successful

    def new_episode(self, args, scene):
        
        if self._env is None:
            if args.arch == 'osx':
                local_executable_path = './datasets/builds/thor-local-OSXIntel64.app/Contents/MacOS/thor-local-OSXIntel64'
            else:
                local_executable_path = './datasets/builds/thor-local-Linux64'
            
            self._env = Environment(
                    grid_size=args.grid_size,
                    fov=args.fov,
                    local_executable_path=local_executable_path,
                    randomize_objects=args.randomize_objects,
                    seed=self.seed)
            self._env.start(scene, self.gpu_id)
        else:
            self._env.reset(scene)

        # For now, single target.
        self.target = ('Tomato', 'Microwave')

        self.open_success = False
        self.put_success = False
        self.close_success = False
        self.tomato_success = False

        self.has_tomato = False
        self.mic = False
        self.tomato_in_mic = False

        self.locate_tomato = 0
        self.open_mic = 0
        self.place_tomato = 0
        self.close_mic = 0

        self.success = False
        self.cur_scene = scene
        self.actions_taken = []
        
        return True
