from pokegym import data

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
MONEY_ADDR_1 = 0xD347
MONEY_ADDR_100 = 0xD348
MONEY_ADDR_10000 = 0xD349

######################################################################################################

def bcd(num):
    return 10 * ((num >> 4) & 0x0F) + (num & 0x0F)

def bit_count(bits):
    return bin(bits).count("1")

def read_bit(game, addr, bit) -> bool:
    return bin(256 + game.get_memory_value(addr))[-bit - 1] == "1"

def mem_val(game, addr):
    mem = game.get_memory_value(addr)
    return mem

def read_uint16(game, start_addr):
    """Read 2 bytes"""
    val_256 = game.get_memory_value(start_addr)
    val_1 = game.get_memory_value(start_addr + 1)
    return 256 * val_256 + val_1

######################################################################################################

def money(game):
    return (100 * 100 * bcd(game.get_memory_value(MONEY_ADDR_1))
        + 100 * bcd(game.get_memory_value(MONEY_ADDR_100))
        + bcd(game.get_memory_value(MONEY_ADDR_10000)))

def player_poke(game):
    id = game.get_memory_value(0xD014)
    # status = game.get_memory_value(0xD018)
    type_1 = game.get_memory_value(0xD019)
    type_2 = game.get_memory_value(0xD01A)
    level = game.get_memory_value(0xD022)
    max_hp = read_uint16(game, 0xD023)  
    attack = read_uint16(game, 0xD025)
    defense = read_uint16(game, 0xD027)
    speed = read_uint16(game, 0xD029)
    special = read_uint16(game, 0xD02B)
    if mem_val(game, 0xD057) == 0: # is_in_battle if 1
       return [0,0,0,0,0,0,0,0,0]
    else:
        return [id, type_1, type_2, level, max_hp, attack, defense, speed, special] #  status,

def op_poke(game):
    id = game.get_memory_value(0xCFE5)
    # status = game.get_memory_value(0xCFE9)
    type_1 = game.get_memory_value(0xCFEA)
    type_2 = game.get_memory_value(0xCFEB)
    level = game.get_memory_value(0xCFF3)
    max_hp = read_uint16(game, 0xCFF4)  
    attack = read_uint16(game, 0xCFF6)
    defense = read_uint16(game, 0xCFF8)
    speed = read_uint16(game, 0xCFFA)
    special = read_uint16(game, 0xCFFC)
    if mem_val(game, 0xD057) == 0: # is_in_battle if 1
       return [0,0,0,0,0,0,0,0,0]
    else:
        return [id, type_1, type_2, level, max_hp, attack, defense, speed, special] #  status,
   
def read_pokemon(game, start_addr):
    type_dict = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            7: 6,
            8: 7,
            20: 8,
            21: 9,
            22: 10,
            23: 11,
            24: 12,
            25: 13,
            26: 14,
            }
    poke_id = game.get_memory_value(start_addr)
    if poke_id == 0:
        poke_type_1 = 0
    else:
        raw_type1 = game.get_memory_value(start_addr + 0x05)
        poke_type_1 = type_dict[raw_type1]
    # poke_type_2 = game.get_memory_value(start_addr + 0x06)
    return poke_id, poke_type_1 #, poke_type_2

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

def position(game):
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
    return game.get_memory_value(WCUTTILE)

def write_mem(game, addr, value):
    mem = game.set_memory_value(addr, value)
    return mem

def badges(game):
    badges = game.get_memory_value(BADGE_1_ADDR)
    return bit_count(badges)

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













