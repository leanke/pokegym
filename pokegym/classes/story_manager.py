from pokegym import ram_map

EVENT = 5

class Story:
    def __init__(self, game):
        self.game = game
        self.tower = [142, 143, 144, 145, 146, 147, 148]
        self.hideout = [199, 200, 201, 202, 203, 135] # includes game corner
        self.silphco = [181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236]
        self.celadon = [134, 6] # gym and celadon
        self.towns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rew_sum = 0
        self.story_maps ={
            0: [], # fossil TODO
            1: [], # bill done TODO
            2: [], # ssanne done TODO
            3: self.hideout + self.celadon, # hideout done
            4: self.tower + [149], # fiji done
            5: self.silphco # silphco done
            # 6: self. # flute item check
        }
    @property
    def story_prog(self):
        fossil = 0
        if int(ram_map.read_bit(self.game, 0xD7F6, 6)) == 1 or int(ram_map.read_bit(self.game, 0xD7F6, 7)) == 1:
            fossil = 1
        return [
        int(fossil), # either dome or helix fossil
        int(ram_map.read_bit(self.game, 0xD7F2, 7)), # left_bills_house_after_helping
        int(ram_map.read_bit(self.game, 0xD803, 5)), # Walked_Out_Of_Dock
        int(ram_map.read_bit(self.game, 0xD81B, 7)), # beat_rocket_hideout_giovanni
        int(ram_map.read_bit(self.game, 0xD769, 7)), # rescued_mr_fuji_2
        int(ram_map.read_bit(self.game, 0xD838, 7)), # Beat_Silph_Co_Giovanni
        # int(ram_map.read_bit(self.game, 0xD76C, 0)), # got_poke_flute
        # int(ram_map.read_bit(self.game, 0xD75F, 0)), # got_bicycle
        # int(ram_map.read_bit(self.game, 0xD771, 1)), # got_bike_voucher
        ]
    
    def maps(self):
        high = set()
        low = set()
        for i in range(len(self.story_prog)):
            if not self.story_prog[i] and self.story_prog[(i-1)]:
                high = self.story_maps[i]
            elif self.story_prog[i]:
                low.add(self.story_maps[i])
            else:
                low.add(self.story_maps[i])
        return high, low
    
    def events(self):
        return sum(self.story_prog)
    
    def update(self):
        self.rew_sum = self.events()

