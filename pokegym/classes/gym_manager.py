from pokegym import ram_map

FILL = 1

TASK = 2
TRAINER = 2
LEADER= 5

class Gym:
    def __init__(self, game):
        self.game = game
        self.maps_left = []
        self.trainer_rew = 0
        self.task_rew = 0
        self.leader_rew = 0
        self.rew_sum = 0
        self.gym_maps = {
            0: 54, # pewter
            1: 65, # Cerulean
            2: 92, # vermilion
            3: 134, # celadon
            4: 157, # Fuchsia
            5: 178, # Saffron
            6: 166, # Cinnabar
            7: 45, # viridian
        }
        self.gym_trainers = {
            0:{0xD755: [2]},
            1:{0xD75E: [2, 3]},
            2:{0xD773: [2, 3, 4]},
            3:{0xD77C: [2, 3, 4, 5, 6, 7], 0xD77D: [0]},
            4:{0xD792: [2, 3, 4, 5, 6, 7]},
            5:{0xD7B3: [2, 3, 4, 5, 6, 7], 0xD7B4: [0]},
            6:{0xD79A: [2, 3, 4, 5, 6, 7], 0xD79B: [0]},
            7:{0xD751: [2, 3, 4, 5, 6, 7], 0xD752: [0, 1]},
        }
        self.gym_tasks = {
            # 0:{FILL: [1]},
            # 1:{FILL: [1]},
            2:{0xD773: [0, 1]},
            # 3:{FILL: [1]},
            # 4:{FILL: [1]},
            # 5:{FILL: [1]},
            # 6:{FILL: [1]},
            7:{0xD74C: [0]},
        }
        self.gym_leaders = {
            0:{0xD755: [7]},
            1:{0xD75E: [7]},
            2:{0xD773: [7]},
            3:{0xD77C: [1]},
            4:{0xD792: [1]},
            5:{0xD7B3: [1]},
            6:{0xD79A: [1]},
            7:{0xD751: [1]},
        }


    @property
    def gym_prog(self): 
        # checks the gym leader bits
        return [
        int(ram_map.read_bit(self.game, 0xD755, 7)), # Pewter
        int(ram_map.read_bit(self.game, 0xD75E, 7)), # Cerulean
        int(ram_map.read_bit(self.game, 0xD773, 7)), # Vermilion
        int(ram_map.read_bit(self.game, 0xD77C, 1)), # Celadon
        int(ram_map.read_bit(self.game, 0xD792, 1)), # Fuchsia
        int(ram_map.read_bit(self.game, 0xD7B3, 1)), # Saffron
        int(ram_map.read_bit(self.game, 0xD79A, 1)), # Cinnabar
        int(ram_map.read_bit(self.game, 0xD751, 1))] # Viridian
    
    def maps(self):
        high = set()
        low = set()
        for i in range(len(self.gym_prog)):
            if self.gym_prog[i]:
                low.add(self.gym_maps[i])
            else:
                high.add(self.gym_maps[i])
        return high, low
    
    def events(self, dictionary):
        perc_list = [0] * 8
        counter = 0
        for key, value in dictionary.items():
            # complete = 0
            # count = 0
            for k, v in value.items():
                for i in range(len(v)):
                    # count += 1
                    if int(ram_map.read_bit(self.game, k, v[i])):
                        counter += 1
                        # complete += 1
            # perc_list[int(key)] = (complete/count) # this is a percentage of tasks done 
        return counter
    
    def update(self):
        self.trainer_rew = (self.events(self.gym_trainers) * TRAINER)
        self.task_rew = (self.events(self.gym_tasks) * TASK)
        self.leader_rew = (self.events(self.gym_leaders) * LEADER)
        self.rew_sum = (self.trainer_rew + self.task_rew + self.leader_rew)
    