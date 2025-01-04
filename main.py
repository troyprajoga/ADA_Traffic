import pygame
import sys
import time
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
PINK = (255, 105, 180)
PURPLE = (128, 0, 128)
obstacle_color = (169, 169, 169)

# Global Variables
grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
nodes = []  # List of nodes (manually defined)
road_segments = []  # Store all road segments with unique properties
start = None
end = None
road_counter = 0
current_obstacles = []  # To track the currently displayed obstacles
original_tile_states = {}  # Dictionary to store the original state of tiles before placing obstacles
obstacles = [(4, 4), (7, 15), (11, 3), (15, 8), (1, 21), (16, 17), (20, 10)]  # Fixed obstacle positions
final_path = []
optimized_final_path = []
trail_visible = False
optimized_trail_visible = False


def draw_grid():
    """Draw the grid and fill the cells with appropriate colors."""
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE
            if grid[row][col] == 1:
                color = BLACK  # Default road color
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

def place_fixed_obstacles():
    """
    Place fixed obstacles at predefined positions.
    Obstacles are marked as -1 in the grid and shown as gray dots.
    """
    for obstacle in obstacles:
        row, col = obstacle
        grid[row][col] = -1  # Mark as an obstacle

def remove_all_obstacles():
    """Remove all obstacles from the grid."""
    global current_obstacles
    for obstacle in current_obstacles:
        row, col = obstacle
        grid[row][col] = original_tile_states.get((row, col), 0)
    current_obstacles = []



def draw_obstacles():
    """Draw currently active obstacles."""
    for obstacle in current_obstacles:
        row, col = obstacle
        pygame.draw.rect(screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.circle(screen, obstacle_color, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)



def randomize_obstacles(num_obstacles):
    """Randomize the placement of obstacles."""
    global current_obstacles, original_tile_states
    for obstacle in current_obstacles:
        row, col = obstacle
        grid[row][col] = original_tile_states.get((row, col), 0)
    current_obstacles = random.sample(obstacles, min(num_obstacles, len(obstacles)))
    for obstacle in current_obstacles:
        row, col = obstacle
        original_tile_states[(row, col)] = grid[row][col]
        grid[row][col] = -1



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
    """Perform Depth-First Search (DFS) to find a path."""
    start_time = time.perf_counter()
    stack = [start]
    visited = set()
    parent = {}
    tiles_traveled = 0

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        tiles_traveled += 1
        if current == end:
            path = []
            while current:
                path.append(current)
                current = parent.get(current)
            execution_time = (time.perf_counter() - start_time) * 1_000_000
            return path[::-1], tiles_traveled, execution_time
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS and grid[neighbor[0]][neighbor[1]] == 1:
                if neighbor not in visited:
                    stack.append(neighbor)
                    parent[neighbor] = current

    execution_time = (time.perf_counter() - start_time) * 1_000_000
    return None, tiles_traveled, execution_time




def highlight_path(path, color):
    """Highlight the final path on the grid."""
    for node in path:
        if node == start or node == end:
            continue
        row, col = node
        pygame.draw.circle(screen, color, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)


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

def redraw():
    """Redraw the entire screen."""
    screen.fill(WHITE)
    draw_grid()
    draw_obstacles()
    if trail_visible:
        highlight_path(final_path, PINK)
    if optimized_trail_visible:
        highlight_path(optimized_final_path, PURPLE)
    pygame.display.flip()


def toggle_path():
    """Toggle visibility of the regular DFS path."""
    global trail_visible
    trail_visible = not trail_visible
    redraw()


def visualize_trail(trail, start, end):
    """Visualize the trail of the search process."""
    for node in trail:
        if node == start or node == end:
            continue
        row, col = node
        pygame.draw.circle(screen, CYAN, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)
        pygame.display.flip()
        pygame.time.delay(50)

def calculate_path_weight(path):
    """
    Calculate the weighted cost of the path and a breakdown of tile counts by type.
    :param path: List of nodes representing the path.
    :return: Tuple (total_weight, tile_breakdown).
    """
    total_weight = 0
    tile_breakdown = {"black": 0, "yellow": 0, "orange": 0, "red": 0}

    for node in path:
        row, col = node
        weight = 1.0  # Default weight for black tiles
        if grid[row][col] == 1:  # Black tile
            tile_breakdown["black"] += 1
        for segment in road_segments:
            if (row, col) in segment["cells"]:
                weight = segment["weight"]
                if weight == 1.3:
                    tile_breakdown["yellow"] += 1
                elif weight == 1.6:
                    tile_breakdown["orange"] += 1
                elif weight == 2.0:
                    tile_breakdown["red"] += 1
                break
        total_weight += weight

    return total_weight, tile_breakdown


def dfs_with_heuristics(grid, start, end):
    """Perform an optimized DFS with heuristics."""
    start_time = time.perf_counter()
    stack = [(start, 0)]
    visited = set()
    parent = {}
    tiles_traveled = 0

    def heuristic(node):
        return abs(node[0] - end[0]) + abs(node[1] - end[1])

    while stack:
        stack.sort(key=lambda x: x[1] + heuristic(x[0]), reverse=True)
        current, current_cost = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        tiles_traveled += 1
        if current == end:
            path = []
            while current:
                path.append(current)
                current = parent.get(current)
            execution_time = (time.perf_counter() - start_time) * 1_000_000
            return path[::-1], tiles_traveled, execution_time
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < ROWS and 0 <= neighbor[1] < COLS and grid[neighbor[0]][neighbor[1]] == 1:
                if neighbor not in visited:
                    stack.append((neighbor, current_cost + 1))
                    parent[neighbor] = current

    execution_time = (time.perf_counter() - start_time) * 1_000_000
    return None, tiles_traveled, execution_time


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
    """Toggle visibility of the optimized DFS path."""
    global optimized_trail_visible
    optimized_trail_visible = not optimized_trail_visible
    redraw()


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

# Main Loop
running = True
while running:
    for event in pygame.event.get():
        # Quit the game
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()

        # Set start node on left-click
        elif pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            row, col = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
            if (row, col) in nodes:
                start = (row, col)
                print(f"Start node set to: {start}")

        # Set end node on right-click
        elif pygame.mouse.get_pressed()[2]:
            pos = pygame.mouse.get_pos()
            row, col = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
            if (row, col) in nodes:
                end = (row, col)
                print(f"End node set to: {end}")

        # Handle keyboard events
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:  # Randomize 1 obstacle
                randomize_obstacles(1)
                redraw()

            elif event.key == pygame.K_2:  # Randomize 2 obstacles
                randomize_obstacles(2)
                redraw()

            elif event.key == pygame.K_3:  # Randomize 3 obstacles
                randomize_obstacles(3)
                redraw()

            elif event.key == pygame.K_y:  # Remove all obstacles
                remove_all_obstacles()
                redraw()

            elif event.key == pygame.K_SPACE and start and end:  # Regular DFS
                print(f"Running DFS from {start} to {end}")
                new_path, tiles_traveled, exec_time_microseconds = dfs(grid, start, end)

                if new_path:
                    print(f"Path found in {exec_time_microseconds:.2f} μs")
                    print(f"Tiles traveled: {tiles_traveled}, Path length: {len(new_path)}")
                    final_path = new_path
                    total_weight, tile_breakdown = calculate_path_weight(new_path)
                    print(f"Path weight: {total_weight:.2f}, Breakdown: {tile_breakdown}")

                    visualize_trail(new_path, start, end)
                    highlight_path(final_path, (255, 105, 180))  # Pink
                    trail_visible = True
                else:
                    print("No path found!")

                redraw()

            elif event.key == pygame.K_r:  # Toggle DFS path visibility
                toggle_path()

            elif event.key == pygame.K_o and start and end:  # Optimized DFS
                print(f"Running Optimized DFS from {start} to {end}")
                new_optimized_path, tiles_traveled, exec_time_microseconds = dfs_with_heuristics(grid, start, end)

                if new_optimized_path:
                    print(f"Optimized path found in {exec_time_microseconds:.2f} μs")
                    print(f"Tiles traveled: {tiles_traveled}, Path length: {len(new_optimized_path)}")
                    optimized_final_path = new_optimized_path
                    total_weight, tile_breakdown = calculate_path_weight(new_optimized_path)
                    print(f"Path weight: {total_weight:.2f}, Breakdown: {tile_breakdown}")

                    visualize_trail(new_optimized_path, start, end)
                    highlight_path(optimized_final_path, (128, 0, 128))  # Purple
                    optimized_trail_visible = True
                else:
                    print("No optimized path found!")

                redraw()

            elif event.key == pygame.K_p:  # Toggle optimized DFS path visibility
                toggle_optimized_path()

    # Redraw the screen
    redraw()
    pygame.display.flip()
