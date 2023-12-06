import numpy as np

class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.map = np.zeros((height, width), dtype=np.uint8)
        self.visited_value = 80

    def visit_tile(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.map[y, x] = self.visited_value

    def print_map(self):
        print(self.map)

# Example usage:
map = Map(width=16, height=16)

# Visiting a few tiles
map.visit_tile(3, 4)
map.visit_tile(7, 8)
map.visit_tile(12, 2)

# Print the updated map
game.print_map()

# In this example:

#     The Game class has a NumPy array (self.map) representing the game map. 
#     The array is initialized with zeros.

#     The visited_tile method is used to mark a tile as visited by updating 
#     the corresponding value in the array to self.visited_value (which is set to 80).

#     The print_map method is just for visualization purposes, to see the state of the map.

# You can adapt this structure to your specific needs, incorporating it into your existing 
# codebase or game representation. The key idea is to use a 2D NumPy array to represent the 
# game map and update the values as you visit different tiles.
