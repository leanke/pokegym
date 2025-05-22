

class Story:
    def __init__(self, events):
        self.events = events
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
        self.events.bit_check('')
        fossil = 0
        if self.events.bit_check('Got_Dome_Fossil') or self.events.bit_check('Got_Helix_Fossil'):
            fossil = 1
        return [
        int(fossil), # either dome or helix fossil
        self.events.bit_check('left_bills_house_after_helping'), # left_bills_house_after_helping
        self.events.bit_check('beat_rocket_hideout_giovanni'), # Walked_Out_Of_Dock
        self.events.bit_check('beat_rocket_hideout_giovanni'), # beat_rocket_hideout_giovanni
        self.events.bit_check('rescued_mr_fuji_2'), # rescued_mr_fuji_2
        self.events.bit_check('Beat_Silph_Co_Giovanni'), # Beat_Silph_Co_Giovanni
        # int(ram_map.read_bit(self.game, 0xD76C, 0)), # got_poke_flute
        # int(ram_map.read_bit(self.game, 0xD75F, 0)), # got_bicycle
        # int(ram_map.read_bit(self.game, 0xD771, 1)), # got_bike_voucher
        ]
    
    def maps(self):
        high = []
        low = []
        for i in range(len(self.story_prog)):
            if not self.story_prog[i] and self.story_prog[(i-1)]:
                high = self.story_maps[i]
            elif self.story_prog[i]:
                low += self.story_maps[i]
            else:
                low += self.story_maps[i]
        return high, low
    
        
