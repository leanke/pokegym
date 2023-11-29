# Destination data

DEST1 = 0xD401 # 00 - overworld, 01 - dungeon, 02 - side view area 
DEST2 = 0xD402 # Values from 00 to 1F accepted. FF is Color Dungeon 
DEST3 = 0xD403 # Room number. Must appear on map or it will lead to an empty room 
XMAP = 0xD404 # X map coord
YMAP = 0xD405 # Y map coord
LOADED_MAP = range(0xD700, 0xD79B) # Currently loaded map

'''
World map status

Each screen status is represented by a byte, which is a combination of the following masks :
00 : Unexplored
10 : changed from initial status (for example sword taken on the beach or dungeon opened with key)
20 : owl talked
80 : visited
A0 : visited and owl
B0 : visited, owl, and changed
For example, visiting the first dungeon's screen (80) and opening it with the key (10) would put that byte at 90 
'''
MAP_STATUS = range(0xD800, 0xD8FF) # World map status

# Your currently held items. 
SLOTB = 0xDB00
SLOTA = 0xDB01
 # Inventory
INV0 = 0xDB02
INV1 = 0xDB03
INV2 = 0xDB04
INV3 = 0xDB05
INV4 = 0xDB06
INV5 = 0xDB07
INV6 = 0xDB08
INV7 = 0xDB09
INV8 = 0xDB0A
INV9 = 0xDB0B
# Items IDs
SWORD = 0x01
BOMBS = 0x02
POWER_BRACELET = 0x03
SHIELD = 0x04
BOW = 0x05
HOOKSHOT = 0x06
FIRE_ROD = 0x07
PEGASUS_BOOTS = 0x08
OCARINA = 0x09
FEATHER = 0x0A
SHOVEL = 0x0B
MAGIC_POWDER = 0x0C
BOOMERANG = 0x0D

# Dung Keys
KEY_START = 0xDB10
KEY2 = 0xDB11
KEY3 = 0xDB12
KEY4 = 0xDB13
KEY_STOP = 0xDB14

INTRO = 0xDB97

FLIPPERS = 0xDB0C # Flippers (01=have)
POTION = 0xDB0D # Potion (01=have) 
ITEM_TRADE = 0xDB0E # Current item in trading game (01=Yoshi, 0E=magnifier) 
SHELLS = 0xDB0F # Number of secret shells 
LEAVES = 0xDB15 # Number of golden leaves 
DUNG_ITEM_FLAG = range(0xDB16, 0xDB3D) # Beginning of dungeon item flags. 5 bytes fo each dungeon, 5th byte is quantity of keys for that dungeon
BRACELET = 0xDB43 # Power bracelet level 
SHEILD = 0xDB44 # Shield level 
NUM_ARROW = 0xDB45 # Number of arrows 
SONGS = 0xDB49 # Ocarina songs in possession (3 bits mask, 0=no songs, 7=all songs)
SELECT_SONG = 0xDB4A # Ocarina selected song
NUM_POWDER = 0xDB4C # Magic powder quantity 
NUM_BOMB = 0xDB4D # Number of bombs 
SWORD_LEVEL = 0xDB4E # Sword level 

# Number of times the character died for each save slot (one byte per save slot) 
DEATH2 = 0xDB56 
DEATH1 = 0xDB57
DEATH3 = 0xDB58

HEART = 0xDB5A # Current health. Each increment of 08h is one full heart, each increment of 04h is one-half heart
MAX_HEART = 0xDB5B # Maximum health. Simply counts the number of hearts Link has in hex. Max recommended value is 0Eh (14 hearts)
# Number of rupees (for 999 put 0999) 
NUM_RUPEE1 = 0xDB5D
NUM_RUPEE2 = 0xDB5E
INSTRUMENT = range(0xDB65, 0xDB6C) # Instruments for every dungeon, 00=no instrument, 03=have instrument 
MAX_POWDER = 0xDB76 # Max magic powder
MAX_BOMB = 0xDB77 # Max bombs 
MAX_ARROW = 0xDB78 # Max arrows 
DUNG_COORD = 0xDBAE # Your position on the 8x8 dungeon grid
NUM_KEY = 0xDBD0 # Quantity of keys in posession 
MAPTILE = 0xDB54

ITEMS_MAP = {
    0x01: 'SWORD',
    0x02: 'BOMBS',
    0x03: 'POWER_BRACELET',
    0x04: 'SHIELD',
    0x05: 'BOW',
    0x06: 'HOOKSHOT',
    0x07: 'FIRE_ROD',
    0x08: 'PEGASUS_BOOTS',
    0x09: 'OCARINA',
    0x0A: 'FEATHER',
    0x0B: 'SHOVEL',
    0x0C: 'MAGIC_POWDER',
    0x0D: 'BOOMERANG',
}

INV_ADDRESSES = [
    INV0, INV1, INV2, INV3, INV4,
    INV5, INV6, INV7, INV8, INV9,
]

DUNG_KEYS = [
    KEY_START, KEY2, KEY3, KEY4, KEY_STOP
]

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
    x_pos = game.get_memory_value(XMAP)
    y_pos = game.get_memory_value(YMAP)
    tile = game.get_memory_value(MAPTILE)
    return x_pos, y_pos, tile


#new functions

def hp_fraction(game):
    hp_sum = game.get_memory_value(HEART)
    max_hp_sum = game.get_memory_value(MAX_HEART)
    if max_hp_sum == 0:
        return 0
    return max_hp_sum * (hp_sum / (8 * max_hp_sum)) # matches hearts half heart = .5

def intro(game):
    byte = game.get_memory_value(INTRO)
    if byte == 198:
         return 1
    return 0

def read_inv(game):
    inventory = []
    for address in INV_ADDRESSES:
        item_id = game.get_memory_value(address)
        item_name = ITEMS_MAP.get(item_id, 'UNKNOWN')
        inventory.append((address, item_name))
    return inventory

def read_rupees(game):
    return (100 * bcd(game.get_memory_value(NUM_RUPEE1))
    + bcd(game.get_memory_value(NUM_RUPEE2)))

def death_count(game):
    death1 = bit_count(game.get_memory_value(DEATH1))
    death2 = bit_count(game.get_memory_value(DEATH2))
    death3 = bit_count(game.get_memory_value(DEATH3))
    return death1

def secret_shell(game):
    shell = game.get_memory_value(SHELLS)
    return shell

def dung_keys(game):
    keys = sum(bit_count(game.get_memory_value(i)) for i in range(KEY_START, KEY_STOP))
    return keys

def read_held_items(game):
    slot_a = game.get_memory_value(SLOTA)
    slot_b = game.get_memory_value(SLOTB)
    return slot_a, slot_b

def dest_status(game):
    dest1_status = {
        0x00: 'dungeon',
        0x01: 'Overworld',
        0x02: 'SideView',
        0xFF: 'Forest?',
        }
    dest1 = game.get_memory_value(DEST1)
    status = dest1_status.get(dest1, 'Unknown')
    if dest1 == 0:
        return status, 1
    return status, 0
