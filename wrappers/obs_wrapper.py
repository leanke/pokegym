import numpy as np
from typing import List, Tuple
from gymnasium import Env, spaces
import gymnasium as gym

# add self.add_boey_obs to your env to activate/deactivate the boey_obs
# in this file you can choose pyboy_version 1 or 2 *untested if this works*
# in torch.py there are two functions boey_obs and boey_nets that will have to be added to a policy


class ObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.time = env.time
        self.n_pokemon_features = 23
        self.pokecenter_ids = [0x01, 0x02, 0x03, 0x0F, 0x15, 0x05, 0x06, 0x04, 0x07, 0x08, 0x0A]
        self.output_vector_shape = (54, )
        self.visited_pokecenter_list = []
        self.last_10_map_ids = np.zeros(10, dtype=np.uint8)
        self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        self.init_caches()
        self.past_events_string = ''
        self.last_10_event_ids = np.zeros((10, 2), dtype=np.float32)
        self.boey_obs = {
                    'vector': spaces.Box(low=-1, high=1, shape=self.output_vector_shape, dtype=np.float32),
                    'map_ids': spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
                    'item_ids': spaces.Box(low=0, high=255, shape=(20,), dtype=np.uint8),
                    'item_quantity': spaces.Box(low=-1, high=1, shape=(20, 1), dtype=np.float32),
                    'poke_ids': spaces.Box(low=0, high=255, shape=(12,), dtype=np.uint8),
                    'poke_type_ids': spaces.Box(low=0, high=255, shape=(12, 2), dtype=np.uint8),
                    'poke_move_ids': spaces.Box(low=0, high=255, shape=(12, 4), dtype=np.uint8),
                    'poke_move_pps': spaces.Box(low=-1, high=1, shape=(12, 4, 2), dtype=np.float32),
                    'poke_all': spaces.Box(low=-1, high=1, shape=(12, self.n_pokemon_features), dtype=np.float32),
                    'event_ids': spaces.Box(low=0, high=2570, shape=(10,), dtype=np.uint32),
                    'event_step_since': spaces.Box(low=-1, high=1, shape=(10, 1), dtype=np.float32),
                }
        if self.env.add_boey_obs:
            self.observation_space = spaces.Dict(self.env.observation_space, **self.boey_obs)
        else:
            self.observation_space = self.env.observation_space

        self.pyboy_version = 1 # set to 1 for pyboy <2.0.0 and set to 2 for pyboy >=2.0.0
        if hasattr(env, "pyboy"):
            self.pyboy = env.pyboy
        elif hasattr(env, "game"):
            self.pyboy = env.game
        else:
            raise Exception("Could not find emulator!")
        
    def _get_obs(self):
        if self.env.add_boey_obs:
            return { # self.env._get_obs().update(
                'vector': self.get_all_raw_obs(),
                'map_ids': self.get_last_map_id_obs(),
                'item_ids': self.get_all_item_ids_obs(),
                'item_quantity': self.get_items_quantity_obs(),
                'poke_ids': self.get_all_pokemon_ids_obs(),
                'poke_type_ids': self.get_all_pokemon_types_obs(),
                'poke_move_ids': self.get_all_move_ids_obs(),
                'poke_move_pps': self.get_all_move_pps_obs(),
                'poke_all': self.get_all_pokemon_obs(),
                'event_ids': self.get_all_event_ids_obs(),
                'event_step_since': self.get_all_event_step_since_obs(),
            } # )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        full_obs = dict(obs, **self._get_obs())
        self.visited_pokecenter_list = []
        self.last_10_map_ids = np.zeros(10, dtype=np.uint8)
        self.last_10_coords = np.zeros((10, 2), dtype=np.uint8)
        self.init_caches()
        self.past_events_string = ''
        self.last_10_event_ids = np.zeros((10, 2), dtype=np.float32)
        return full_obs, info

    def step(self, action):
        obs, reward, done, done, info = self.env.step(action)
        full_obs = dict(obs, **self._get_obs())
        self.update_last_center()
        self.update_past_events()
        self.init_caches()
        self.past_events_string = self.all_events_string
        return full_obs, reward, done, done, info
    
    def update_last_center(self):
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1 and last_pokecenter_id not in self.visited_pokecenter_list:
            self.visited_pokecenter_list.append(last_pokecenter_id)

    def multi_hot_encoding(self, cnt, max_n):
        return [1 if cnt < i else 0 for i in range(max_n)]
    
    def one_hot_encoding(self, cnt, max_n, start_zero=False):
        if start_zero:
            return [1 if cnt == i else 0 for i in range(max_n)]
        else:
            return [1 if cnt == i+1 else 0 for i in range(max_n)]
    
    def scaled_encoding(self, cnt, max_n: float):
        max_n = float(max_n)
        if isinstance(cnt, list):
            return [min(1.0, c / max_n) for c in cnt]
        elif isinstance(cnt, np.ndarray):
            return np.clip(cnt / max_n, 0, 1)
        else:
            return min(1.0, cnt / max_n)
    
    def get_badges_obs(self):
        return self.multi_hot_encoding(self.get_badges(), 12)

    def get_money_obs(self):
        return [self.scaled_encoding(self.read_money(), 100_000)]
    
    def read_swap_mon_pos(self):
        is_in_swap_mon_party_menu = self.read_m(0xd07d) == 0x04
        if is_in_swap_mon_party_menu:
            chosen_mon = self.read_m(0xcc35)
            if chosen_mon == 0:
                print(f'\nsomething went wrong, chosen_mon is 0')
            else:
                return chosen_mon - 1
        return -1
    
    def get_last_pokecenter_obs(self):
        return self.get_last_pokecenter_list()

    def get_visited_pokecenter_obs(self):
        result = [0] * len(self.pokecenter_ids)
        for i in self.visited_pokecenter_list:
            result[i] = 1
        return result
    
    def get_hm_move_obs(self):
        hm_moves = [0x0f, 0x13, 0x39, 0x46, 0x94]
        result = [0] * len(hm_moves)
        all_moves = self.get_party_moves()
        for i, hm_move in enumerate(hm_moves):
            if hm_move in all_moves:
                result[i] = 1
                continue
        return result
    
    def get_hm_obs(self):
        hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
        items = self.get_items_in_bag()
        result = [0] * len(hm_ids)
        for i, hm_id in enumerate(hm_ids):
            if hm_id in items:
                result[i] = 1
                continue
        return result
    
    def get_items_in_bag(self, one_indexed=0):
        first_item = 0xD31E
        # total 20 items
        # item1, quantity1, item2, quantity2, ...
        item_ids = []
        for i in range(0, 20, 2):
            item_id = self.read_m(first_item + i)
            if item_id == 0 or item_id == 0xff:
                break
            item_ids.append(item_id + one_indexed)
        return item_ids
    
    def get_items_obs(self):
        # items from self.get_items_in_bag()
        # add 0s to make it 20 items
        items = self.get_items_in_bag(one_indexed=1)
        items.extend([0] * (20 - len(items)))
        return items

    def get_items_quantity_obs(self):
        # items from self.get_items_quantity_in_bag()
        # add 0s to make it 20 items
        items = self.get_items_quantity_in_bag()
        items = self.scaled_encoding(items, 20)
        items.extend([0] * (20 - len(items)))
        return np.array(items, dtype=np.float32).reshape(-1, 1)

    def get_bag_full_obs(self):
        # D31D
        return [1 if self.read_m(0xD31D) >= 20 else 0]
    
    def get_last_10_map_ids_obs(self):
        return self.last_10_map_ids
    
    def get_last_10_coords_obs(self):
        # 10, 2
        # scale x with 45, y with 72
        result = []
        for coord in self.last_10_coords:
            result.append(min(coord[0] / 45, 1))
            result.append(min(coord[1] / 72, 1))
        return result
    
    def get_pokemon_ids_obs(self):
        return self.read_party(one_indexed=1)
    
    def read_party(self, one_indexed=0):
        parties = [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]
        return [p + one_indexed if p != 0xff and p != 0 else 0 for p in parties]
    
    def get_battle_pokemon_ids_obs(self):
        battle_pkmns = [self.read_m(addr) for addr in [0xcfe5, 0xd014]]
        return [p + 1 if p != 0xff and p != 0 else 0 for p in battle_pkmns]
    
    def get_party_types_obs(self):
        # 6 pokemon, 2 types each
        # start from D170 type1, D171 type2
        # next pokemon will be + 44
        # 0xff is no pokemon
        result = []
        for i in range(0, 44*6, 44):
            # 2 types per pokemon
            type1 = self.read_m(0xD170 + i)
            type2 = self.read_m(0xD171 + i)
            result.append(type1)
            result.append(type2)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_opp_types_obs(self):
        # 6 pokemon, 2 types each
        # start from D8A9 type1, D8AA type2
        # next pokemon will be + 44
        # 0xff is no pokemon
        result = []
        for i in range(0, 44*6, 44):
            # 2 types per pokemon
            type1 = self.read_m(0xD8A9 + i)
            type2 = self.read_m(0xD8AA + i)
            result.append(type1)
            result.append(type2)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_battle_types_obs(self):
        # CFEA type1, CFEB type2
        # d019 type1, d01a type2
        result = [self.read_m(0xcfea), self.read_m(0xCFEB), self.read_m(0xD019), self.read_m(0xD01A)]
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_party_move_ids_obs(self):
        # D173 move1, D174 move2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            moves = [self.read_m(addr + i) for addr in [0xD173, 0xD174, 0xD175, 0xD176]]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_opp_move_ids_obs(self):
        # D8AC move1, D8AD move2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            moves = [self.read_m(addr + i) for addr in [0xD8AC, 0xD8AD, 0xD8AE, 0xD8AF]]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_battle_move_ids_obs(self):
        # CFED move1, CFEE move2
        # second pokemon starts from D003
        result = []
        for addr in [0xCFED, 0xD003]:
            moves = [self.read_m(addr + i) for i in range(4)]
            result.extend(moves)
        return [p + 1 if p != 0xff and p != 0 else 0 for p in result]
    
    def get_party_move_pps_obs(self):
        # D188 pp1, D189 pp2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            pps = [self.read_m(addr + i) for addr in [0xD188, 0xD189, 0xD18A, 0xD18B]]
            result.extend(pps)
        return result
    
    def get_opp_move_pps_obs(self):
        # D8C1 pp1, D8C2 pp2...
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            pps = [self.read_m(addr + i) for addr in [0xD8C1, 0xD8C2, 0xD8C3, 0xD8C4]]
            result.extend(pps)
        return result
    
    def get_battle_move_pps_obs(self):
        # CFFE pp1, CFFF pp2
        # second pokemon starts from D02D
        result = []
        for addr in [0xCFFE, 0xD02D]:
            pps = [self.read_m(addr + i) for i in range(4)]
            result.extend(pps)
        return result
    
    def get_party_level_obs(self):
        # D18C level
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            level = self.read_m(0xD18C + i)
            result.append(level)
        return result
    
    def get_opp_level_obs(self):
        # D8C5 level
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            level = self.read_m(0xD8C5 + i)
            result.append(level)
        return result
    
    def get_battle_level_obs(self):
        # CFF3 level
        # second pokemon starts from D037
        result = []
        for addr in [0xCFF3, 0xD022]:
            level = self.read_m(addr)
            result.append(level)
        return result
    
    def get_all_level_obs(self):
        result = []
        result.extend(self.get_party_level_obs())
        result.extend(self.get_opp_level_obs())
        result.extend(self.get_battle_level_obs())
        result = np.array(result, dtype=np.float32) / 100
        # every elemenet max is 1
        result = np.clip(result, 0, 1)
        return result
    
    def get_party_hp_obs(self):
        # D16C hp
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            hp = self.read_hp(0xD16C + i)
            max_hp = self.read_hp(0xD18D + i)
            result.extend([hp, max_hp])
        return result
    
    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    def get_opp_hp_obs(self):
        # D8A5 hp
        # next pokemon will be + 44
        result = []
        for i in range(0, 44*6, 44):
            hp = self.read_hp(0xD8A5 + i)
            max_hp = self.read_hp(0xD8C6 + i)
            result.extend([hp, max_hp])
        return result
    
    def get_battle_hp_obs(self):
        # CFE6 hp
        # second pokemon starts from CFFC
        result = []
        for addr in [0xCFE6, 0xCFF4, 0xCFFC, 0xD00A]:
            hp = self.read_hp(addr)
            result.append(hp)
        return result
    
    def get_all_hp_obs(self):
        result = []
        result.extend(self.get_party_hp_obs())
        result.extend(self.get_opp_hp_obs())
        result.extend(self.get_battle_hp_obs())
        result = np.array(result, dtype=np.float32)
        # every elemenet max is 1
        result = np.clip(result, 0, 600) / 600
        return result
    
    def get_all_hp_pct_obs(self):
        hps = []
        hps.extend(self.get_party_hp_obs())
        hps.extend(self.get_opp_hp_obs())
        hps.extend(self.get_battle_hp_obs())
        # divide every hp by max hp
        hps = np.array(hps, dtype=np.float32)
        hps = hps.reshape(-1, 2)
        hps = hps[:, 0] / (hps[:, 1] + 0.00001)
        # every elemenet max is 1
        return hps
    
    def get_all_pokemon_dead_obs(self):
        # 1 if dead, 0 if alive
        hp_pct = self.get_all_hp_pct_obs()
        return [1 if hp <= 0 else 0 for hp in hp_pct]
    
    def get_battle_status_obs(self):
        # D057
        # 0 not in battle return 0, 0
        # 1 wild battle return 1, 0
        # 2 trainer battle return 0, 1
        # -1 lost battle return 0, 0
        result = []
        status = self.battle_type
        if status == 1:
            result = [1, 0]
        elif status == 2:
            result = [0, 1]
        else:
            result = [0, 0]
        return result
    
    def fix_pokemon_type(self, ptype: int) -> int:
        if ptype < 9:
            return ptype
        elif ptype < 27:
            return ptype - 11
        else:
            print(f'invalid pokemon type: {ptype}')
            return 16
        
    def get_pokemon_types(self, start_addr):
        return [self.fix_pokemon_type(self.read_m(start_addr + i)) + 1 for i in range(2)]
        
    def get_all_pokemon_types_obs(self):
        # 6 party pokemon types start from D170
        # 6 enemy pokemon types start from D8A9
        party_type_addr = 0xD170
        enemy_type_addr = 0xD8A9
        result = []
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            # 2 types per pokemon
            ptypes = self.get_pokemon_types(party_type_addr + i * 44)
            result.append(ptypes)
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append([0, 0])
        if self.is_in_battle():
            # zero padding if not in battle, reduce dimension
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    # 2 types per pokemon
                    ptypes = self.get_pokemon_types(enemy_type_addr + i * 44)
                    result.append(ptypes)
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0, 0])
            else:
                wild_ptypes = self.get_pokemon_types(0xCFEA)  # 2 ptypes only, add padding for remaining 5
                result.append(wild_ptypes)
                result.extend([[0, 0]] * 5)
        else:
            result.extend([[0, 0]] * 6)
        result = np.array(result, dtype=np.uint8)  # shape (24,)
        assert result.shape == (12, 2), f'invalid ptypes shape: {result.shape}'  # set PYTHONOPTIMIZE=1 to disable assert
        return result
    
    def get_pokemon_status(self, addr):
        # status
        # bit 0 - 6
        # one byte has 8 bits, bit unused: 7
        statuses = [self.read_bit(addr, i) for i in range(7)]
        return statuses  # shape (7,)
    
    def get_one_pokemon_obs(self, start_addr, team, position, is_wild=False):
        # team 0 = my team, 1 = opp team
        # 1 pokemon, address start from start_addr
        # +0 = id
        # +5 = type1 (15 types) (physical 0 to 8 and special 20 to 26)  + 1 to be 1 indexed, 0 is no pokemon/padding
        # +6 = type2 (15 types)
        # +33 = level
        # +4 = status (bit 0-6)
        # +1 = current hp (2 bytes)
        # +34 = max hp (2 bytes)
        # +36 = attack (2 bytes)
        # +38 = defense (2 bytes)
        # +40 = speed (2 bytes)
        # +42 = special (2 bytes)
        # exclude id, type1, type2
        result = []
        # status
        status = self.get_pokemon_status(start_addr + 4)
        result.extend(status)
        # level
        level = self.scaled_encoding(self.read_m(start_addr + 33), 100)
        result.append(level)
        # hp
        hp = self.scaled_encoding(self.read_double(start_addr + 1), 250)
        result.append(hp)
        # max hp
        max_hp = self.scaled_encoding(self.read_double(start_addr + 34), 250)
        result.append(max_hp)
        # attack
        attack = self.scaled_encoding(self.read_double(start_addr + 36), 134)
        result.append(attack)
        # defense
        defense = self.scaled_encoding(self.read_double(start_addr + 38), 180)
        result.append(defense)
        # speed
        speed = self.scaled_encoding(self.read_double(start_addr + 40), 140)
        result.append(speed)
        # special
        special = self.scaled_encoding(self.read_double(start_addr + 42), 154)
        result.append(special)
        # is alive
        is_alive = 1 if hp > 0 else 0
        result.append(is_alive)
        # is in battle, check position 0 indexed against the following addr
        if is_wild:
            in_battle = 1
        else:
            if self.is_in_battle():
                if team == 0:
                    in_battle = 1 if position == self.read_m(0xCC35) else 0
                else:
                    in_battle = 1 if position == self.read_m(0xCFE8) else 0
            else:
                in_battle = 0
        result.append(in_battle)
        # my team 0 / opp team 1
        result.append(team)
        # position 0 to 5, one hot, 5 elements, first pokemon is all 0
        result.extend(self.one_hot_encoding(position, 5))
        # is swapping this pokemon
        if team == 0:
            swap_mon_pos = self.read_swap_mon_pos()
            if swap_mon_pos != -1:
                is_swapping = 1 if position == swap_mon_pos else 0
            else:
                is_swapping = 0
        else:
            is_swapping = 0
        result.append(is_swapping)
        return result

    def get_party_pokemon_obs(self):
        # 6 party pokemons start from D16B
        # 2d array, 6 pokemons, N features
        result = np.zeros((6, self.n_pokemon_features), dtype=np.float32)
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result[i] = self.get_one_pokemon_obs(0xD16B + i * 44, 0, i)
        for i in range(pokemon_count, 6):
            result[i] = np.zeros(self.n_pokemon_features, dtype=np.float32)
        return result

    def read_opp_pokemon_num(self):
        return self.read_m(0xD89C)
    
    def get_battle_base_pokemon_obs(self, start_addr, team):
        # CFE5
        result = []
        # status
        status = self.get_pokemon_status(start_addr + 4)
        result.extend(status)
        # level
        level = self.scaled_encoding(self.read_m(start_addr + 14), 100)
        result.append(level)
        # hp
        hp = self.scaled_encoding(self.read_double(start_addr + 1), 250)
        result.append(hp)
        # max hp
        max_hp = self.scaled_encoding(self.read_double(start_addr + 15), 250)
        result.append(max_hp)
        # attack
        attack = self.scaled_encoding(self.read_double(start_addr + 17), 134)
        result.append(attack)
        # defense
        defense = self.scaled_encoding(self.read_double(start_addr + 19), 180)
        result.append(defense)
        # speed
        speed = self.scaled_encoding(self.read_double(start_addr + 21), 140)
        result.append(speed)
        # special
        special = self.scaled_encoding(self.read_double(start_addr + 23), 154)
        result.append(special)
        # is alive
        is_alive = 1 if hp > 0 else 0
        result.append(is_alive)
        # is in battle, check position 0 indexed against the following addr
        in_battle = 1
        result.append(in_battle)
        # my team 0 / opp team 1
        result.append(team)
        # position 0 to 5, one hot, 5 elements, first pokemon is all 0
        result.extend(self.one_hot_encoding(0, 5))
        return result
    
    def get_wild_pokemon_obs(self):
        start_addr = 0xCFE5
        return self.get_battle_base_pokemon_obs(start_addr, team=1)

    def get_opp_pokemon_obs(self):
        # 6 enemy pokemons start from D8A4
        # 2d array, 6 pokemons, N features
        result = []
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result.append(self.get_one_pokemon_obs(0xD8A4 + i * 44, 1, i))
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0] * self.n_pokemon_features)
            else:
                # wild battle, take the battle pokemon
                result.append(self.get_wild_pokemon_obs())
                for i in range(5):
                    result.append([0] * self.n_pokemon_features)
        else:
            return np.zeros((6, self.n_pokemon_features), dtype=np.float32)
        result = np.array(result, dtype=np.float32)
    
    def get_all_pokemon_obs(self):
        # 6 party pokemons start from D16B
        # 6 enemy pokemons start from D8A4
        # gap between each pokemon is 44
        party = self.get_party_pokemon_obs()
        opp = self.get_opp_pokemon_obs()
        # print(f'party shape: {party.shape}, opp shape: {opp.shape}')
        result = np.concatenate([party, opp], axis=0)
        return result  # shape (12, 22)
    
    def get_party_pokemon_ids_obs(self):
        # 6 party pokemons start from D16B
        # 1d array, 6 pokemons, 1 id
        result = []
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result.append(self.read_m(0xD16B + i * 44) + 1)
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append(0)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def get_opp_pokemon_ids_obs(self):
        # 6 enemy pokemons start from D8A4
        # 1d array, 6 pokemons, 1 id
        result = []
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result.append(self.read_m(0xD8A4 + i * 44) + 1)
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append(0)
            else:
                # wild battle, take the battle pokemon
                result.append(self.read_m(0xCFE5) + 1)
                for i in range(5):
                    result.append(0)
        else:
            return np.zeros(6, dtype=np.uint8)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def get_all_pokemon_ids_obs(self):
        # 6 party pokemons start from D16B
        # 6 enemy pokemons start from D8A4
        # gap between each pokemon is 44
        party = self.get_party_pokemon_ids_obs()
        opp = self.get_opp_pokemon_ids_obs()
        result = np.concatenate((party, opp), axis=0)
        return result
    
    def get_one_pokemon_move_ids_obs(self, start_addr):
        # 4 moves
        return [self.read_m(start_addr + i) for i in range(4)]
    
    def get_party_pokemon_move_ids_obs(self):
        # 6 party pokemons start from D173
        # 2d array, 6 pokemons, 4 moves
        result = []
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result.append(self.get_one_pokemon_move_ids_obs(0xD173 + (i * 44)))
        remaining_pokemon = 6 - pokemon_count
        for i in range(remaining_pokemon):
            result.append([0] * 4)
        result = np.array(result, dtype=np.uint8)
        return result

    def get_opp_pokemon_move_ids_obs(self):
        # 6 enemy pokemons start from D8AC
        # 2d array, 6 pokemons, 4 moves
        result = []
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result.append(self.get_one_pokemon_move_ids_obs(0xD8AC + (i * 44)))
                remaining_pokemon = 6 - pokemon_count
                for i in range(remaining_pokemon):
                    result.append([0] * 4)
            else:
                # wild battle, take the battle pokemon
                result.append(self.get_one_pokemon_move_ids_obs(0xCFED))
                for i in range(5):
                    result.append([0] * 4)
        else:
            return np.zeros((6, 4), dtype=np.uint8)
        result = np.array(result, dtype=np.uint8)
        return result
    
    def get_all_move_ids_obs(self):
        # 6 party pokemons start from D173
        # 6 enemy pokemons start from D8AC
        # gap between each pokemon is 44
        party = self.get_party_pokemon_move_ids_obs()
        opp = self.get_opp_pokemon_move_ids_obs()
        result = np.concatenate((party, opp), axis=0)
        return result  # shape (12, 4)
    
    def get_one_pokemon_move_pps_obs(self, start_addr):
        # 4 moves
        result = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            pp = self.scaled_encoding(self.read_m(start_addr + i), 30)
            have_pp = 1 if pp > 0 else 0
            result[i] = [pp, have_pp]
        return result
    
    def get_party_pokemon_move_pps_obs(self):
        # 6 party pokemons start from D188
        # 2d array, 6 pokemons, 8 features
        # features: pp, have pp
        result = np.zeros((6, 4, 2), dtype=np.float32)
        pokemon_count = self.read_num_poke()
        for i in range(pokemon_count):
            result[i] = self.get_one_pokemon_move_pps_obs(0xD188 + (i * 44))
        for i in range(pokemon_count, 6):
            result[i] = np.zeros((4, 2), dtype=np.float32)
        return result
    
    def get_opp_pokemon_move_pps_obs(self):
        # 6 enemy pokemons start from D8C1
        # 2d array, 6 pokemons, 8 features
        # features: pp, have pp
        result = np.zeros((6, 4, 2), dtype=np.float32)
        if self.is_in_battle():
            if not self.is_wild_battle():
                pokemon_count = self.read_opp_pokemon_num()
                for i in range(pokemon_count):
                    result[i] = self.get_one_pokemon_move_pps_obs(0xD8C1 + (i * 44))
                for i in range(pokemon_count, 6):
                    result[i] = np.zeros((4, 2), dtype=np.float32)
            else:
                # wild battle, take the battle pokemon
                result.append(self.get_one_pokemon_move_pps_obs(0xCFFE))
                for i in range(5):
                    result.append(np.zeros((4, 2), dtype=np.float32))
        else:
            return np.zeros((6, 4, 2), dtype=np.float32)
        return result
    
    def get_all_move_pps_obs(self):
        # 6 party pokemons start from D188
        # 6 enemy pokemons start from D8C1
        party = self.get_party_pokemon_move_pps_obs()
        opp = self.get_opp_pokemon_move_pps_obs()
        result = np.concatenate((party, opp), axis=0)
        return result
    
    def get_all_item_ids_obs(self):
        # max 85
        return np.array(self.get_items_obs(), dtype=np.uint8)
    
    def get_all_event_ids_obs(self):
        # max 249
        # padding_idx = 0
        # change dtype to uint8 to save space
        return np.array(self.last_10_event_ids[:, 0] + 1, dtype=np.uint8)
    
    def get_all_event_step_since_obs(self):
        step_gotten = self.last_10_event_ids[:, 1]  # shape (10,)
        step_since = self.time - step_gotten
        # step_count - step_since and scaled_encoding
        return self.scaled_encoding(step_since, 1000).reshape(-1, 1)  # shape (10,)
    
    def get_last_coords_obs(self):
        # 2 elements
        coord = self.last_10_coords[0]
        return [self.scaled_encoding(coord[0], 45), self.scaled_encoding(coord[1], 72)]
    
    def get_num_turn_in_battle_obs(self):
        if self.is_in_battle:
            return self.scaled_encoding(self.read_m(0xCCD5), 30)
        else:
            return 0
    
    def get_all_raw_obs(self):
        obs = []
        obs.extend(self.get_badges_obs())
        obs.extend(self.get_money_obs())
        obs.extend(self.get_last_pokecenter_obs())
        obs.extend(self.get_visited_pokecenter_obs())
        obs.extend(self.get_hm_move_obs())
        obs.extend(self.get_hm_obs())
        obs.extend(self.get_battle_status_obs())
        pokemon_count = self.read_num_poke()
        obs.extend([self.scaled_encoding(pokemon_count, 6)])  # number of pokemon
        obs.extend([1 if pokemon_count == 6 else 0])  # party full
        obs.extend([self.scaled_encoding(self.read_m(0xD31D), 20)])  # bag num items
        obs.extend(self.get_bag_full_obs())  # bag full
        obs.extend(self.get_last_coords_obs())  # last coords x, y
        obs.extend([self.get_num_turn_in_battle_obs()])  # num turn in battle
        # obs.extend(self.get_reward_check_obs())  # reward check
        return np.array(obs, dtype=np.float32)

    def get_last_map_id_obs(self):
        return np.array([self.last_10_map_ids[0]], dtype=np.uint8)
    
    def get_in_battle_mask_obs(self):
        return np.array([self.is_in_battle()], dtype=np.float32)

    def update_past_events(self):
        if self.past_events_string and self.past_events_string != self.all_events_string:
            self.last_10_event_ids = np.roll(self.last_10_event_ids, 1, axis=0)
            self.last_10_event_ids[0] = [self.get_first_diff_index(self.past_events_string, self.all_events_string), self.time]

    def read_num_poke(self):
        return self.read_m(0xD163)
    
    def get_items_quantity_in_bag(self):
        first_quantity = 0xD31F
        # total 20 items
        # quantity1, item2, quantity2, ...
        item_quantities = []
        for i in range(1, 20, 2):
            item_quantity = self.read_m(first_quantity + i)
            if item_quantity == 0 or item_quantity == 0xff:
                break
            item_quantities.append(item_quantity)
        return item_quantities
    
    def is_in_battle(self):
        # D057
        # 0 not in battle
        # 1 wild battle
        # 2 trainer battle
        # -1 lost battle
        return self.battle_type > 0
    
    @property
    def battle_type(self):
        if not self._battle_type:
            result = self.read_m(0xD057)
            if result == -1:
                return 0
            return result
        return self._battle_type
    
    def is_wild_battle(self):
        return self.battle_type == 1
    
    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))
    
    def read_money(self):
        return (100 * 100 * self.read_bcd(self.read_m(0xD347)) + 
                100 * self.read_bcd(self.read_m(0xD348)) +
                self.read_bcd(self.read_m(0xD349)))
    
    def get_last_pokecenter_list(self):
        pc_list = [0, ] * len(self.pokecenter_ids)
        last_pokecenter_id = self.get_last_pokecenter_id()
        if last_pokecenter_id != -1:
            pc_list[last_pokecenter_id] = 1
        return pc_list
    
    def get_last_pokecenter_id(self):
        last_pokecenter = self.read_m(0xD719)
        # will throw error if last_pokecenter not in pokecenter_ids, intended
        if last_pokecenter == 0:
            # no pokecenter visited yet
            return -1
        return self.pokecenter_ids.index(last_pokecenter)
    
    def get_party_moves(self):
        # first pokemon moves at D173
        # 4 moves per pokemon
        # next pokemon moves is 44 bytes away
        first_move = 0xD173
        moves = []
        for i in range(0, 44*6, 44):
            # 4 moves per pokemon
            move = [self.read_m(first_move + i + j) for j in range(4)]
            moves.extend(move)
        return moves
    
    def read_m(self, addr):
        if self.pyboy_version == 1:
            return self.pyboy.get_memory_value(addr)
        if self.pyboy_version == 2:
            return self.pyboy.memory[addr]

    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    def read_double(self, start_add):
        return 256*self.read_m(start_add) + self.read_m(start_add+1)
    
    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    @property
    def all_events_string(self):
        # cache all events string to improve performance
        if not self._all_events_string:
            event_flags_start = 0xD747
            event_flags_end = 0xD886
            result = ''
            for i in range(event_flags_start, event_flags_end):
                result += bin(self.read_m(i))[2:]  # .zfill(8)
            self._all_events_string = result
        return self._all_events_string
    
    def init_caches(self):
        # for cached properties
        self._all_events_string = ''
        self._battle_type = -999

    def get_first_diff_index(self, arr1, arr2):
        for i in range(len(arr1)):
            if arr1[i] != arr2[i]:
                return i
        return -1