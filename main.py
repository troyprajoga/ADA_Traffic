import pygame
import sys
import random

pygame.init()
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 25, 25  # 25x25 grid
CELL_SIZE = WIDTH // COLS
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Map Pathfinder with Traffic Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)

final_path = []  # Store the most recent final path
trail_visible = False  # Track whether the path is visible

optimized_final_path = []  # Store the most recent final path for the optimized DFS
optimized_trail_visible = False  # Track visibility of the optimized path

# Grid and other variables
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
nodes = []  # List of nodes (manually defined)
road_segments = []  # Store all road segments with unique properties
start = None
end = None
road_counter = 0  # Global counter for unique road IDs

def draw_grid():
    """Draws the grid and fills the cells with appropriate colors."""
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE
            if grid[row][col] == 1:
                color = BLACK  # Default road color
                # Check if the cell is part of a road segment with traffic
                for segment in road_segments:
                    if (row, col) in segment["cells"]:
                        weight = segment["weight"]
                        if weight == 1.3:
                            color = YELLOW
                        elif weight == 1.6:
                            color = ORANGE
                        elif weight == 2.0:
                            color = RED
            pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, BLUE, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

    # Draw nodes
    for node in nodes:
        if node == start:
            color = GREEN  # Start node
        elif node == end:
            color = RED  # End node
        else:
            color = CYAN  # Default node color (visible until selected)
        x, y = node[1] * CELL_SIZE + CELL_SIZE // 2, node[0] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, color, (x, y), CELL_SIZE // 4)

    # Draw start and end points
    if start:
        x, y = start[1] * CELL_SIZE + CELL_SIZE // 2, start[0] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, GREEN, (x, y), CELL_SIZE // 3)
    if end:
        x, y = end[1] * CELL_SIZE + CELL_SIZE // 2, end[0] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, RED, (x, y), CELL_SIZE // 3)

def build_road(start_pos, direction, length):
    global road_counter
    segment_id = f"Road_{road_counter}"
    road_counter += 1

    row, col = start_pos
    segment = {"id": segment_id, "cells": [], "weight": 1.0}  # Default weight is 1
    for _ in range(length):
        if 0 <= row < ROWS and 0 <= col < COLS:
            grid[row][col] = 1
            segment["cells"].append((row, col))  # Add cell to this road segment
        row += direction[0]
        col += direction[1]
    road_segments.append(segment)  # Store the road segment

def build_staircase_diagonal(start_pos, length, direction="down-right"):
    global road_counter
    segment_id = f"Road_{road_counter}"
    road_counter += 1

    row, col = start_pos
    segment = {"id": segment_id, "cells": [], "weight": 1.0}  # Default weight is 1
    for _ in range(length):
        if 0 <= row < ROWS and 0 <= col < COLS:
            grid[row][col] = 1
            segment["cells"].append((row, col))  # Add cell to this road segment
        if direction == "down-right":
            if _ % 2 == 0:
                row += 1  # Move downward
            else:
                col += 1  # Move right
        elif direction == "up-right":
            if _ % 2 == 0:
                row -= 1  # Move upward
            else:
                col += 1  # Move right
    road_segments.append(segment)  # Store the road segment

def build_single_road(position):
    """
    Marks a single grid cell as a traversable road.
    :param position: Tuple (row, col) specifying the grid cell.
    """
    row, col = position
    if 0 <= row < ROWS and 0 <= col < COLS:  # Ensure it's within grid bounds
        grid[row][col] = 1  # Mark as road

def assign_traffic_weights():
    """
    Assigns traffic weights to 20% of the road segments to simulate traffic.
    Roads with traffic are colored:
    - Yellow (1.3x weight)
    - Orange (1.6x weight)
    - Red (2.0x weight)
    """
    # Traffic levels and their weights
    traffic_weights = {
        "yellow": 1.3,
        "orange": 1.6,
        "red": 2.0
    }
    # Select approximately 20% of the road segments
    num_traffic_roads = max(1, int(0.2 * len(road_segments)))  # At least 1 road gets traffic
    traffic_segments = random.sample(road_segments, num_traffic_roads)

    for segment in traffic_segments:
        # Randomly assign a traffic level
        traffic_level = random.choice(["yellow", "orange", "red"])
        segment["weight"] = traffic_weights[traffic_level]

        # Update the grid colors for this segment
        for cell in segment["cells"]:
            row, col = cell
            grid[row][col] = 1  # Keep it traversable


def dfs(grid, start, end):
    """
    Perform Depth-First Search (DFS) to find a path from start to end.
    :param grid: 2D grid representing the map.
    :param start: Tuple (row, col) for the starting position.
    :param end: Tuple (row, col) for the ending position.
    :return: Tuple (path, tiles_traveled) where:
        - path: List of nodes representing the path, or None if no path is found.
        - tiles_traveled: Total number of tiles visited during traversal.
    """
    stack = [start]  # Use a stack to explore nodes (LIFO)
    visited = set()  # Track visited nodes
    parent = {}  # Keep track of the path
    tiles_traveled = 0  # Count total tiles visited

    while stack:
        current = stack.pop()
        if current in visited:
            continue

        visited.add(current)
        tiles_traveled += 1

        # Visualize the search process
        row, col = current
        pygame.draw.circle(screen, CYAN, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)
        pygame.display.flip()
        pygame.time.delay(50)

        # Check if we reached the end
        if current == end:
            path = []
            while current:
                path.append(current)
                current = parent.get(current)
            return path[::-1], tiles_traveled  # Reverse the path and return tiles traveled

        # Explore neighbors (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)

            # Check if the neighbor is within bounds and is a road
            if 0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS and grid[neighbor[0]][neighbor[1]] == 1:
                if neighbor not in visited:
                    stack.append(neighbor)
                    parent[neighbor] = current  # Record the parent for backtracking

    return None, tiles_traveled  # No path found



def highlight_path(path):
    """
    Highlights the path found by DFS on the grid.
    :param path: List of nodes representing the path.
    """
    for node in path:
        if node == start or node == end:
            continue
        row, col = node
        pygame.draw.circle(screen, (255, 105, 180), (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)
        pygame.display.flip()
        pygame.time.delay(50)

def clear_final_path():
    """Clears the final path from the grid."""
    for node in final_path:
        row, col = node
        pygame.draw.rect(screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.flip()

def draw_trail():
    """
    Draws a trail of yellow nodes for the final path.
    """
    for node in final_path:
        row, col = node
        pygame.draw.circle(screen, YELLOW, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)
    pygame.display.flip()

def toggle_path():
    """
    Toggles the visibility of the final path.
    """
    global trail_visible  # Declare trail_visible as global
    trail_visible = not trail_visible  # Toggle the trail visibility

    if trail_visible:  # If the path should be visible, draw it
        for node in final_path:
            row, col = node
            pygame.draw.circle(screen, (255, 105, 180), (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)
    else:  # If the path should be hidden, redraw the grid without the path
        screen.fill(WHITE)  # Clear the screen
        draw_grid()  # Redraw the grid and nodes
        if optimized_trail_visible:  # Ensure the optimized path is visible if it's toggled on
            for node in optimized_final_path:
                row, col = node
                pygame.draw.circle(screen, (128, 0, 128), (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)

    pygame.display.flip()  # Update the display




def dfs_with_heuristics(grid, start, end):
    """
    Perform an optimized DFS using weights and heuristics to prioritize traversal.
    :param grid: 2D grid representing the map.
    :param start: Tuple (row, col) for the starting position.
    :param end: Tuple (row, col) for the ending position.
    :return: Tuple (path, tiles_traveled) where:
        - path: List of nodes representing the path, or None if no path is found.
        - tiles_traveled: Total number of tiles visited during traversal.
    """
    stack = [(start, 0)]  # Store (node, cumulative_cost) in stack
    visited = set()  # Track visited nodes
    parent = {}  # Keep track of the path
    tiles_traveled = 0  # Count total tiles visited

    def heuristic(node):
        """Calculate Manhattan distance from the current node to the endpoint."""
        return abs(node[0] - end[0]) + abs(node[1] - end[1])

    while stack:
        # Sort the stack to prioritize nodes with the lowest weight + heuristic
        stack.sort(key=lambda x: x[1] + heuristic(x[0]), reverse=True)
        current, current_cost = stack.pop()  # Pop the node with the lowest cost + heuristic

        if current in visited:
            continue

        visited.add(current)
        tiles_traveled += 1

        # Visualize the search process
        row, col = current
        pygame.draw.circle(screen, CYAN, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)
        pygame.display.flip()
        pygame.time.delay(50)

        # Check if we reached the end
        if current == end:
            path = []
            while current:
                path.append(current)
                current = parent.get(current)
            return path[::-1], tiles_traveled  # Reverse the path and return tiles traveled

        # Explore neighbors (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)

            # Check if the neighbor is within bounds and is a road
            if 0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS and grid[neighbor[0]][neighbor[1]] == 1:
                if neighbor not in visited:
                    # Determine the weight of the road segment the neighbor belongs to
                    weight = 1.0  # Default weight
                    for segment in road_segments:
                        if neighbor in segment["cells"]:
                            weight = segment["weight"]
                            break

                    # Add the neighbor to the stack with the updated cumulative cost
                    stack.append((neighbor, current_cost + weight))
                    parent[neighbor] = current  # Record the parent for backtracking

    return None, tiles_traveled  # No path found


def highlight_optimized_path(path):
    """
    Highlights the path found by the optimized DFS on the grid.
    :param path: List of nodes representing the path.
    """
    for node in path:
        if node == start or node == end:
            continue
        row, col = node
        pygame.draw.circle(screen, (128, 0, 128), (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)
        pygame.display.flip()
        pygame.time.delay(50)


def toggle_optimized_path():
    """
    Toggles the visibility of the optimized DFS final path.
    """
    global optimized_trail_visible  # Declare optimized_trail_visible as global
    optimized_trail_visible = not optimized_trail_visible  # Toggle the trail visibility

    if optimized_trail_visible:  # If the path should be visible, draw it
        for node in optimized_final_path:
            row, col = node
            pygame.draw.circle(screen, (128, 0, 128), (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)
    else:  # If the path should be hidden, redraw the grid without the path
        screen.fill(WHITE)  # Clear the screen
        draw_grid()  # Redraw the grid and nodes
        if trail_visible:  # Ensure the regular path is visible if it's toggled on
            for node in final_path:
                row, col = node
                pygame.draw.circle(screen, (255, 105, 180), (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)

    pygame.display.flip()  # Update the display


def calculate_final_path_weight(path):
    """
    Calculate the weighted cost of the final path based on tile colors (weights).
    :param path: List of nodes representing the final path.
    :return: The total weighted cost of the path and a breakdown of tiles by weight.
    """
    total_weighted_cost = 0
    color_count = {"black": 0, "yellow": 0, "orange": 0, "red": 0}

    for node in path:
        # Skip the start and end nodes, as they do not contribute to the cost
        if node == start or node == end:
            continue

        weight = 1.0  # Default weight for black tiles
        color = "black"

        # Check the weight/color of the current node
        for segment in road_segments:
            if node in segment["cells"]:
                weight = segment["weight"]
                if weight == 1.3:
                    color = "yellow"
                elif weight == 1.6:
                    color = "orange"
                elif weight == 2.0:
                    color = "red"
                break

        # Increment the count for this color
        color_count[color] += 1
        # Add the weight to the total cost
        total_weighted_cost += weight

    return total_weighted_cost, color_count

def create_fixed_map():
    single_road_positions = [
        (23, 10), (12, 8), (16, 8), (12, 12), (16, 12),
        (23, 23), (1, 7), (7, 1), (7, 7), (4, 4),
        (9, 15), (1, 23), (9, 23), (11, 22)
        
    ]

    for position in single_road_positions:
        build_single_road(position)
    # Build roads
    build_road((1, 2), (0, 1), 5)
    build_road((7, 2), (0, 1), 5)
    build_road((2, 1), (1, 0), 5)
    build_road((2, 7), (1, 0), 5)
    build_road((4, 5), (0, 1), 2)
    build_road((4, 2), (0, 1), 2)
    build_road((2, 4), (1, 0), 2)
    build_road((5, 4), (1, 0), 2)
    build_road((1, 8), (0, 1), 5)
    build_staircase_diagonal((1, 13), 5, "down-right")
    build_road((7, 8), (0, 1), 2)
    build_staircase_diagonal((7, 10), 4, "up-right")
    build_road((2, 11), (1, 0), 3)
    build_road((4, 15), (1, 0), 5)
    build_road((9, 16), (0, 1), 7)
    build_staircase_diagonal((4, 16), 7, "up-right")
    build_road((1, 20), (0, 1), 3)
    build_road((2, 23), (1, 0), 7)
    build_staircase_diagonal((4, 16), 8, "down-right")
    build_road((12, 9), (0, 1), 3)
    build_road((13, 8), (1, 0), 3)
    build_road((16, 9), (0, 1), 3)
    build_road((13, 12), (1, 0), 3)
    build_staircase_diagonal((8, 5), 7, "down-right")
    build_road((14, 13), (0, 1), 10)
    build_road((10, 19), (1, 0), 4)
    build_road((11, 20), (0, 1), 2)
    build_road((12, 22), (1, 0), 2)
    build_road((15, 23), (1, 0), 8)
    build_road((23, 11), (0, 1), 12)
    build_road((17, 10), (1, 0), 6)
    build_staircase_diagonal((18, 11), 8, "down-right")
    build_staircase_diagonal((15, 17), 7, "down-right")
    build_road((19, 20), (0, 1), 3)
    build_road((7, 2), (1, 0), 4)
    build_road((10, 1), (0, 1), 3)
    build_road((11, 1), (1, 0), 7)
    build_road((11, 3), (1, 0), 10)
    build_staircase_diagonal((18, 1), 3, "down-right")
    build_road((21, 4), (0, 1), 6)

    global nodes
    nodes = [
        (1,1), (14,23),(21,3)
    ]

    for node in nodes:
        grid[node[0]][node[1]] = 1

create_fixed_map()
assign_traffic_weights()

running = True
while running:
    screen.fill(WHITE)  # Always clear the screen
    draw_grid()  # Redraw the grid

    # Redraw the regular DFS path if it's toggled on
    if trail_visible:
        for node in final_path:
            row, col = node
            pygame.draw.circle(screen, (255, 105, 180), (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)

    # Redraw the optimized DFS path if it's toggled on
    if optimized_trail_visible:
        for node in optimized_final_path:
            row, col = node
            pygame.draw.circle(screen, (128, 0, 128), (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
        elif pygame.mouse.get_pressed()[0]:  # Left-click to set start
            pos = pygame.mouse.get_pos()
            row, col = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
            if (row, col) in nodes:  # Check if the clicked cell is a node
                start = (row, col)
                print(f"Start node set to: {start}")
        elif pygame.mouse.get_pressed()[2]:  # Right-click to set end
            pos = pygame.mouse.get_pos()
            row, col = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
            if (row, col) in nodes:  # Check if the clicked cell is a node
                end = (row, col)
                print(f"End node set to: {end}")
        elif event.type == pygame.KEYDOWN:  # Handle key presses
            if event.key == pygame.K_SPACE and start and end:  # Spacebar to trigger regular DFS
                print(f"Running DFS from {start} to {end}")
                new_path, tiles_traveled = dfs(grid, start, end)  # Call the regular DFS function

                if new_path:
                    print("Path found!")
                    print(f"Total tiles traveled: {tiles_traveled}")
                    print(f"Final path length (nodes): {len(new_path)}")

                    # Calculate the weighted cost of the final path
                    final_path_cost, color_breakdown = calculate_final_path_weight(new_path)
                    print(f"Final path weighted cost: {final_path_cost:.2f}")
                    print(f"Color breakdown: {color_breakdown}")

                    # Highlight the new path and set it as the final path
                    final_path = new_path  # Update the final path
                    trail_visible = True  # Automatically show the path
                else:
                    print("No path found!")
                    print(f"Total tiles traveled: {tiles_traveled}")
            elif event.key == pygame.K_o and start and end:  # "O" key to trigger optimized DFS
                print(f"Running optimized DFS from {start} to {end}")
                new_optimized_path, tiles_traveled = dfs_with_heuristics(grid, start, end)  # Call the optimized DFS function

                if new_optimized_path:
                    print("Optimized path found!")
                    print(f"Total tiles traveled: {tiles_traveled}")
                    print(f"Optimized final path length (nodes): {len(new_optimized_path)}")

                    # Calculate the weighted cost of the optimized final path
                    optimized_final_path_cost, optimized_color_breakdown = calculate_final_path_weight(new_optimized_path)
                    print(f"Optimized final path weighted cost: {optimized_final_path_cost:.2f}")
                    print(f"Optimized color breakdown: {optimized_color_breakdown}")

                    # Highlight the new path and set it as the optimized final path
                    optimized_final_path = new_optimized_path  # Update the optimized final path
                    optimized_trail_visible = True  # Automatically show the path
                else:
                    print("No optimized path found!")
                    print(f"Total tiles traveled: {tiles_traveled}")
            elif event.key == pygame.K_r:  # Toggle regular DFS path visibility
                toggle_path()

            elif event.key == pygame.K_p:  # Toggle optimized DFS path visibility
                toggle_optimized_path()

    pygame.display.flip()




