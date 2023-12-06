# addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
# https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
HP_ADDR =  [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
PARTY_SIZE_ADDR = 0xD163
PARTY_ADDR = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
POKE_XP_ADDR = [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
CAUGHT_POKE_ADDR = range(0xD2F7, 0xD309)
SEEN_POKE_ADDR = range(0xD30A, 0xD31D)
OPPONENT_LEVEL_ADDR = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
BADGE_1_ADDR = 0xD356
OAK_PARCEL_ADDR = 0xD74E
OAK_POKEDEX_ADDR = 0xD74B
OPPONENT_LEVEL = 0xCFF3
ENEMY_POKE_COUNT = 0xD89C
EVENT_FLAGS_START_ADDR = 0xD747
EVENT_FLAGS_END_ADDR = 0xD761
MUSEUM_TICKET_ADDR = 0xD754
MONEY_ADDR_1 = 0xD347
MONEY_ADDR_100 = 0xD348
MONEY_ADDR_10000 = 0xD349

#Trainer Moves/PP counter if 00 then no move is present
P1MOVES = [0xD173, 0xD174, 0xD175, 0xD176]
P2MOVES = [0xD19F, 0xD1A0, 0xD1A1, 0xD1A2]
P3MOVES = [0xD1CB, 0xD1CC, 0xD1CD, 0xD1CE]
P4MOVES = [0xD1F7, 0xD1F8, 0xD1F9, 0xD1FA]
P5MOVES = [0xD223, 0xD224, 0xD225, 0xD226]
P6MOVES = [0xD24F, 0xD250, 0xD251, 0xD252]

P1MOVEPP = [0xD188, 0xD189, 0xD18A, 0xD18B]
P2MOVEPP = [0xD1B4, 0xD1B5, 0xD1B6, 0xD1B7]
P3MOVEPP = [0xD1E0, 0xD1E1, 0xD1E2, 0xD1E3]
P4MOVEPP = [0xD20C, 0xD20D, 0xD20E, 0xD20F]
P5MOVEPP = [0xD238, 0xD239, 0xD23A, 0xD23B]
P6MOVEPP = [0xD264, 0xD265, 0xD266, 0xD267]


def bcd(num):
    return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)

def bit_count(bits):
    return bin(bits).count('1')

def read_bit(game, addr, bit) -> bool:
    # add padding so zero will read '0b100000000' instead of '0b0'
    return bin(256 + game.get_memory_value(addr))[-bit-1] == '1'

def read_uint16(game, start_addr):
    '''Read 2 bytes'''
    val_256 = game.get_memory_value(start_addr)
    val_1 = game.get_memory_value(start_addr + 1)
    return 256*val_256 + val_1

def position(game):
    r_pos = game.get_memory_value(Y_POS_ADDR)
    c_pos = game.get_memory_value(X_POS_ADDR)
    map_n = game.get_memory_value(MAP_N_ADDR)
    return r_pos, c_pos, map_n

#start new functions

def p1_moves(game):
    p1moves = [game.get_memory_value(addr) for addr in P1MOVES]
    p1movepp= [game.get_memory_value(addr) for addr in P1MOVEPP]
    return p1moves, p1movepp

def p2_moves(game):
    p2moves = [game.get_memory_value(addr) for addr in P2MOVES]
    p2movepp= [game.get_memory_value(addr) for addr in P2MOVEPP]
    return p2moves, p2movepp

def p3_moves(game):
    p3moves = [game.get_memory_value(addr) for addr in P3MOVES]
    p3movepp= [game.get_memory_value(addr) for addr in P3MOVEPP]
    return p3moves, p3movepp

def p4_moves(game):
    p4moves = [game.get_memory_value(addr) for addr in P4MOVES]
    p4movepp= [game.get_memory_value(addr) for addr in P4MOVEPP]
    return p4moves, p4movepp

def p5_moves(game):
    p5moves = [game.get_memory_value(addr) for addr in P5MOVES]
    p5movepp= [game.get_memory_value(addr) for addr in P5MOVEPP]
    return p5moves, p5movepp

def p6_moves(game):
    p6moves = [game.get_memory_value(addr) for addr in P6MOVES]
    p6movepp= [game.get_memory_value(addr) for addr in P6MOVEPP]
    return p6moves, p6movepp

# def move_check(game):
    

#end new functions

def party(game):
    party = [game.get_memory_value(addr) for addr in PARTY_ADDR]
    party_size = game.get_memory_value(PARTY_SIZE_ADDR)
    party_levels = [game.get_memory_value(addr) for addr in PARTY_LEVEL_ADDR]
    return party, party_size, party_levels

def opponent(game):
    return [game.get_memory_value(addr) for addr in OPPONENT_LEVEL_ADDR]

def oak_parcel(game):
    return read_bit(game, OAK_PARCEL_ADDR, 1) 

def pokedex_obtained(game):
    return read_bit(game, OAK_POKEDEX_ADDR, 5)
 
def pokemon_seen(game):
    seen_bytes = [game.get_memory_value(addr) for addr in SEEN_POKE_ADDR]
    return sum([bit_count(b) for b in seen_bytes])

def pokemon_caught(game):
    caught_bytes = [game.get_memory_value(addr) for addr in CAUGHT_POKE_ADDR]
    return sum([bit_count(b) for b in caught_bytes])

def hp(game):
    '''Percentage of total party HP'''
    party_hp = [read_uint16(game, addr) for addr in HP_ADDR]
    party_max_hp = [read_uint16(game, addr) for addr in MAX_HP_ADDR]

    # Avoid division by zero if no pokemon
    sum_max_hp = sum(party_max_hp)
    if sum_max_hp == 0:
        return 1

    return sum(party_hp) / sum_max_hp

def money(game):
    return (100 * 100 * bcd(game.get_memory_value(MONEY_ADDR_1))
        + 100 * bcd(game.get_memory_value(MONEY_ADDR_100))
        + bcd(game.get_memory_value(MONEY_ADDR_10000)))

def badges(game):
    badges = game.get_memory_value(BADGE_1_ADDR)
    return bit_count(badges)

def events(game):
    '''Adds up all event flags, exclude museum ticket'''
    num_events = sum(bit_count(game.get_memory_value(i))
        for i in range(EVENT_FLAGS_START_ADDR, EVENT_FLAGS_END_ADDR))
    museum_ticket = int(read_bit(game, MUSEUM_TICKET_ADDR, 0))

    # Omit 13 events by default
    return max(num_events - 13 - museum_ticket, 0)





# pokemon 1
P1 = 0xD16B# - Pokémon (Again)
P1STAT = 0xD16F# - Status (Poisoned, Paralyzed, etc.)
P1T1 = 0xD170# - Type 1
P1T2 = 0xD171# - Type 2
P1LVL = 0xD18C# - Level (actual level)

# Pokémon 2
P2 = 0xD197# - Pokémon
P2STAT = 0xD19B# - Status
P2T1 = 0xD19C# - Type 1
P2T2 = 0xD19D# - Type 2
P2LVL = 0xD1B8# - Level (actual)

# Pokémon 3
P3 = 0xD1C3# - Pokémon
P3STAT = 0xD1C7# - Status
P3T1 = 0xD1C8# - Type 1
P3T2 = 0xD1C9# - Type 2
P3LVL = 0xD1E4# - Level

# Pokémon 4
P4 = 0xD1EF# - Pokémon
P4STAT = 0xD1F3# - Status
P4T1 = 0xD1F4# - Type 1
P4T2 = 0xD1F5# - Type 2
P4LVL = 0xD210# - Level

# Pokémon 5
P5 = 0xD21B# - Pokémon
P5STAT = 0xD21F# - Status
P5T1 = 0xD220# - Type 1
P5T2 = 0xD221# - Type 2
P5LVL = 0xD23C# - Level

# Pokémon 6
P6 = 0xD247# - Pokémon
P6STAT = 0xD24B# - Status
P6T1 = 0xD24C# - Type 1
P6T2 = 0xD24D# - Type 2
P6LVL = 0xD268# - Level


