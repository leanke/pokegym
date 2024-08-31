from pokegym import ram_map

TRAINER = 1
HM = 5
TWO = 2
TEN= 10
FIVE = 5
RIVAL = 3
QUEST = 5


class Event:
    def __init__(self, game):
        self.game = game
        self.trainer_rew = 0
        self.quest_rew = 0
        self.rival_rew = 0
        self.two_rew = 0
        self.five_rew = 0
        self.ten_rew = 0
        self.hm_rew = 0
        self.note =[ 
          #  [0xD790, 6], # safari_game_over
          #  [0xD790, 7], # in_safari_zone
          #  [0xD838, 5], # Got_Master_Ball
  ]
        self.trainer = {
            0xD825: [2, 3, 4, 5], 0xD827: [2, 3], 0xD829: [2, 3, 4], 
            0xD82B: [2, 3, 4, 5], 0xD82D: [6, 7], 0xD82E: [0], 
            0xD82F: [5, 6, 7], 0xD830: [0], 0xD831: [2, 3, 4], 
            0xD833: [2, 3, 4], 0xD835: [1, 2], 0xD837: [4, 5], 
            0xD7D2: [1, 2, 3, 4, 5, 6, 7], 0xD87D: [1, 2, 3, 4, 5, 6, 7], 
            0xD87E: [0], 0xD7FF: [4, 5], 0xD805: [1, 2, 3, 4], 
            0xD807: [1, 2, 3, 4], 0xD809: [1, 2, 3, 4, 5, 6], 
            0xD7F5: [1, 2, 3, 4, 5, 6, 7], 0xD7F6: [1, 2, 3, 4, 5], 
            0xD7C3: [2, 3, 4, 5, 6, 7], 0xD7C4: [0, 1], 0xD7C5: [2], 
            0xD7EF: [1, 2, 3, 4, 5, 6, 7], 0xD7F1: [1, 2, 3, 4, 5, 6, 7], 
            0xD7F2: [0, 1], 0xD7CF: [1, 2, 3, 4, 5, 6, 7], 0xD7D0: [0, 1], 
            0xD7C9: [1, 2, 3, 4, 5, 6], 0xD7D5: [1, 2, 3, 4, 5, 6, 7], 
            0xD7D6: [0, 1, 2], 0xD7CD: [1, 2, 3, 4, 5, 6, 7], 0xD7CE: [0, 1], 
            0xD7D1: [1, 2, 3, 4, 5, 6], 0xD7D7: [2, 3, 4, 5, 6, 7], 0xD7D8: [0], 
            0xD7DF: [1, 2, 3, 4, 5, 6], 0xD7E1: [1, 2, 3, 4, 5, 6, 7], 0xD7E2: [0, 1, 2], 
            0xD7D9: [1, 2, 3, 4, 5, 6, 7], 0xD7DA: [0, 1, 2], 0xD7DB: [1, 2, 3, 4, 5, 6, 7], 
            0xD7DC: [0, 1, 2], 0xD7DD: [1, 2, 3, 4, 5, 6, 7], 0xD7DE: [0, 1, 2], 
            0xD7E3: [1, 2, 3], 0xD7E5: [1, 2, 3, 4, 5, 6, 7], 0xD7E6: [0, 1, 2], 
            0xD7E7: [1, 2, 3, 4, 5, 6, 7], 0xD7E8: [0, 1, 2], 0xD7E9: [1, 2, 3, 4, 5, 6, 7], 
            0xD7EA: [0, 1], 0xD7F3: [2, 3, 4], 0xD75B: [7], 0xD847: [1], 0xD849: [1, 2], 
            0xD84B: [1, 2], 0xD798: [1], 0xD7B1: [2, 3, 4, 5], 0xD765: [1, 2, 3], 
            0xD766: [1, 2, 3], 0xD767: [2, 3, 4, 5], 0xD768: [1, 2, 3], 0xD769: [1, 2, 3]
        }
        self.quest = {
          0xD826: [5, 6], 0xD828: [0, 1], 0xD82A: [0, 1], 0xD82C: [0, 1, 2], 
          0xD82E: [7], 0xD830: [4, 5, 6], 0xD832: [0], 0xD834: [0, 1, 2, 3], 
          0xD836: [0], 0xD838: [0], 0xD74B: [5], 0xD74E: [1, 0], 0xD747: [6], 
          0xD796: [0], 0xD768: [7], 0xD81B: [5, 6]
        }
        self.rival = {
          0xD74B: [3], 0xD7EB: [0, 1, 5, 6, 7], 
          0xD75A: [0], 0xD764: [6, 7], 0xD82F: [0]
        }
        self.two = {
          0xD815: [1, 2, 3, 4, 5], 0xD817: [1], 0xD819: [1, 2], 
          0xD81B: [2, 3, 4], 0xD755: [6], 0xD75E: [6], 0xD777: [0], 
          0xD778: [4, 5, 6, 7], 0xD77C: [0], 0xD792: [0], 0xD773: [6], 
          0xD7BD: [0], 0xD7AF: [0], 0xD7A1: [7], 0xD826: [7], 0xD79A: [0], 
          0xD751: [0], 0xD74C: [1], 0xD7B3: [0], 0xD7D7: [0], 0xD754: [1], 
          0xD771: [1, 6, 7], 0xD77E: [2, 3, 4], 0xD783: [0], 0xD7BF: [0], 
          0xD7D6: [7], 0xD7DD: [0], 0xD7E0: [7], 0xD85F: [1], 0xD769: [7], 
          0xD7C6: [7], 0xD747: [3, 0], 0xD74A: [2, 0, 1], 0xD74B: [7, 1, 2, 0, 4, 6], 
          0xD7EF: [0], 0xD7F0: [1], 0xD7A3: [0, 1, 2], 0xD7B9: [7], 0xD803: [2, 3, 4, 5]
        }
        self.five = {
          0xD81B: [7], 0xD7B1: [1, 6, 7, 0], 
          0xD838: [7], 0xD75F: [0], 0xD803: [1], 
          0xD7F1: [0], 0xD7F2: [3, 4, 5, 6, 7]
        }
        self.ten = {
          0xD7D8: [6, 7], 0xD7E0: [0, 1], 
          0xD78E: [1], 0xD76C: [0], 
          0xD77E: [1]
        }
        self.hm = {
          0xD803: [0], 
          0xD7E0: [6], 
          0xD857: [0], 
          0xD78E: [0], 
          0xD7C2: [0]
        }


    def events(self, dictionary):
        perc_list = [0] * 8
        counter = 0
        for key, value in dictionary.items():
            for i in range(len(value)):
                if int(ram_map.read_bit(self.game, key, value[i])):
                        counter += 1
            # complete = 0
            # count = 0
            # for k, v in value.items():
            #     for i in range(len(v)):
            #         # count += 1
            #         if int(ram_map.read_bit(self.game, k, v[i])):
            #             counter += 1
                        # complete += 1
            # perc_list[int(key)] = (complete/count) # this is a percentage of tasks done 
        return counter

    def update(self):
        self.trainer_rew = (self.events(self.trainer) * TRAINER)
        self.quest_rew = (self.events(self.quest) * QUEST)
        self.rival_rew = (self.events(self.rival) * RIVAL)
        self.two_rew = (self.events(self.two) * TWO)
        self.five_rew = (self.events(self.five) * FIVE)
        self.ten_rew = (self.events(self.ten) * TEN)
        self.hm_rew = (self.events(self.hm) * HM)
        self.rew_sum = (self.trainer_rew + self.quest_rew + self.rival_rew + self.two_rew + self.five_rew + self.ten_rew + self.hm_rew)



































