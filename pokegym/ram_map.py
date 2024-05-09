# ######################################################################################
#                                        Ram_map
# ######################################################################################

# Data Crystal - https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
# No Comments - https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
# Comments - https://github.com/luckytyphlosion/pokered/blob/master/constants/event_constants.asm
from collections import deque
import numpy as np
from pokegym import data

CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
CUT_SEQ = [((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)), ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),]
HP_ADDR = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
PARTY_SIZE_ADDR = 0xD163
PARTY_ADDR = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
BADGE_1_ADDR = 0xD356
WCUTTILE = 0xCD4D # 61 if Cut used; 0 default. resets to default on map_n change or battle.

##########################################################################################

def bcd(num):
    return 10 * ((num >> 4) & 0x0F) + (num & 0x0F)

def bit_count(bits):
    return bin(bits).count("1")

def read_bit(game, addr, bit) -> bool:
    # add padding so zero will read '0b100000000' instead of '0b0'
    return bin(256 + game.get_memory_value(addr))[-bit - 1] == "1"

def mem_val(game, addr):
    mem = game.get_memory_value(addr)
    return mem

def read_uint16(game, start_addr):
    """Read 2 bytes"""
    val_256 = game.get_memory_value(start_addr)
    val_1 = game.get_memory_value(start_addr + 1)
    return 256 * val_256 + val_1

# MISC #####################################################################################################

def get_hm_count(game):
    hm_ids = [0xC4, 0xC5, 0xC6, 0xC7, 0xC8]
    items = get_items_in_bag(game)
    total_hm_cnt = 0
    for hm_id in hm_ids:
        if hm_id in items:
            total_hm_cnt += 1
    return total_hm_cnt * 1

def get_items_in_bag(game, one_indexed=0):
    first_item = 0xD31E
    item_ids = []
    for i in range(0, 40, 2):
        item_id = game.get_memory_value(first_item + i)
        if item_id == 0 or item_id == 0xff:
            break
        item_ids.append(item_id + one_indexed)
    return item_ids

def position(game): # this is [y, x, z]
    r_pos = game.get_memory_value(Y_POS_ADDR)
    c_pos = game.get_memory_value(X_POS_ADDR)
    map_n = game.get_memory_value(MAP_N_ADDR)
    if r_pos >= 443:
        r_pos = 444
    if r_pos <= 0:
        r_pos = 0
    if c_pos >= 443:
        c_pos = 444
    if c_pos <= 0:
        c_pos = 0
    if map_n > 247:
        map_n = 247
    if map_n < -1:
        map_n = -1
    return r_pos, c_pos, map_n

def party(game):
    # party = [game.get_memory_value(addr) for addr in PARTY_ADDR]
    party_size = game.get_memory_value(PARTY_SIZE_ADDR)
    party_levels = [x for x in [game.get_memory_value(addr) for addr in PARTY_LEVEL_ADDR] if x > 0]
    return party_size, party_levels # [x for x in party_levels if x > 0]

def hp(game):
    """Percentage of total party HP"""
    party_hp = [read_uint16(game, addr) for addr in HP_ADDR]
    party_max_hp = [read_uint16(game, addr) for addr in MAX_HP_ADDR]
    # Avoid division by zero if no pokemon
    sum_max_hp = sum(party_max_hp)
    if sum_max_hp == 0:
        return 1
    return sum(party_hp) / sum_max_hp

def used_cut(game):
    if game.get_memory_value(WCUTTILE) == 61:
        write_mem(game, 0xCD4D, 00) # address, byte to write resets tile check
        return True
    else:
        return False

def write_mem(game, addr, value):
    mem = game.set_memory_value(addr, value)
    return mem

def badges(game):
    badges = game.get_memory_value(BADGE_1_ADDR)
    return bit_count(badges)

def update_pokedex(game):
    seen_pokemon = np.zeros(152, dtype=np.uint8)
    caught_pokemon = np.zeros(152, dtype=np.uint8)
    for i in range(0xD30A - 0xD2F7):
        caught_mem = game.get_memory_value(i + 0xD2F7)
        seen_mem = game.get_memory_value(i + 0xD30A)
        for j in range(8):
            caught_pokemon[8*i + j] = 1 if caught_mem & (1 << j) else 0
            seen_pokemon[8*i + j] = 1 if seen_mem & (1 << j) else 0  
    return sum(seen_pokemon), sum(caught_pokemon)

def update_moves_obtained(game):
    # Scan party
    moves_obtained = {}
    cut = 0
    for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
        if game.get_memory_value(i) != 0:
            for j in range(4):
                move_id = game.get_memory_value(i + j + 8)
                if move_id != 0:
                    if move_id != 0:
                        moves_obtained[move_id] = 1
                    if move_id == 15:
                        cut = 1
    # Scan current box (since the box doesn't auto increment in pokemon red)
    num_moves = 4
    box_struct_length = 25 * num_moves * 2
    for i in range(game.get_memory_value(0xda80)):
        offset = i*box_struct_length + 0xda96
        if game.get_memory_value(offset) != 0:
            for j in range(4):
                move_id = game.get_memory_value(offset + j + 8)
                if move_id != 0:
                    moves_obtained[move_id] = 1
    return sum(moves_obtained), cut

# CUT #####################################################################################################

def cut_array(game):
  cut_coords = {}
  cut_tiles = {} # set([])
  cut_state = deque(maxlen=3)
  if mem_val(game, 0xD057) == 0: # is_in_battle if 1
    player_direction = game.get_memory_value(0xC109)
    y, x, map_id = position()  # x, y, map_id
    if player_direction == 0:  # down
        coords = (x, y + 1, map_id)
    if player_direction == 4:
        coords = (x, y - 1, map_id)
    if player_direction == 8:
        coords = (x - 1, y, map_id)
    if player_direction == 0xC:
        coords = (x + 1, y, map_id)
    cut_state.append(
        (
            game.get_memory_value(0xCFC6),
            game.get_memory_value(0xCFCB),
            game.get_memory_value(0xCD6A),
            game.get_memory_value(0xD367),
            game.get_memory_value(0xD125),
            game.get_memory_value(0xCD3D),
        )
    )
    if tuple(list(cut_state)[1:]) in CUT_SEQ:
        cut_coords[coords] = 5 # from 14
        cut_tiles[cut_state[-1][0]] = 1
    elif cut_state == CUT_GRASS_SEQ:
        cut_coords[coords] = 0.001
        cut_tiles[cut_state[-1][0]] = 1
    elif deque([(-1, *elem[1:]) for elem in cut_state]) == CUT_FAIL_SEQ:
        cut_coords[coords] = 0.001
        cut_tiles[cut_state[-1][0]] = 1
    if int(read_bit(game, 0xD803, 0)):
        if check_if_in_start_menu(game):
            seen_start_menu = 1
        if check_if_in_pokemon_menu(game):
            seen_pokemon_menu = 1
        if check_if_in_stats_menu(game):
            seen_stats_menu = 1
        if check_if_in_bag_menu(game):
            seen_bag_menu = 1
  return cut_coords, cut_tiles, seen_start_menu, seen_pokemon_menu, seen_stats_menu, seen_bag_menu

def check_if_in_start_menu(game) -> bool:
    return (
        mem_val(game, 0xD057) == 0
        and mem_val(game, 0xCF13) == 0
        and mem_val(game, 0xFF8C) == 6
        and mem_val(game, 0xCF94) == 0
    )

def check_if_in_pokemon_menu(game) -> bool:
    return (
        mem_val(game, 0xD057) == 0
        and mem_val(game, 0xCF13) == 0
        and mem_val(game, 0xFF8C) == 6
        and mem_val(game, 0xCF94) == 2
    )

def check_if_in_stats_menu(game) -> bool:
    return (
        mem_val(game, 0xD057) == 0
        and mem_val(game, 0xCF13) == 0
        and mem_val(game, 0xFF8C) == 6
        and mem_val(game, 0xCF94) == 1
    )

def check_if_in_bag_menu(game) -> bool:
    return (
        mem_val(game, 0xD057) == 0
        and mem_val(game, 0xCF13) == 0
        # and mem_val(game, 0xFF8C) == 6 # only sometimes
        and mem_val(game, 0xCF94) == 3
    )

