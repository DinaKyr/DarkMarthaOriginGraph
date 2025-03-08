import matplotlib
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


# Load the data
events_df = pd.read_csv('Dark_GD_Contest_Events.csv')
edges_df = pd.read_csv('Dark_GD_Contest_Edges.csv')
cross_img = Image.open("rip-Photoroom.png")
# Origin events and edges
origin_events = events_df[(events_df['World'] == 'Origin') | (events_df['World'] == 'Origin (End)')]
origin_event_ids = origin_events['ID'].astype(float).astype(str).tolist()

# Martha events and edges
martha_events = events_df[events_df['World'] == 'Martha']
martha_event_ids = martha_events['ID'].astype(float).astype(str).tolist()

# Convert Source deaths Target to string
edges_df['Source'] = edges_df['Source'].astype(str)
edges_df['Target'] = edges_df['Target'].astype(str)
death_nodes = events_df[events_df["Death"] == True]["ID"].astype(str).tolist()
important_nodes = events_df[(events_df["Important Trigger"] == True) & (events_df["World"].isin(["Origin", "Martha"]))]["ID"].astype(str).tolist()


# Origin only edges
origin_edges = edges_df[edges_df['Source'].isin(origin_event_ids) & edges_df['Target'].isin(origin_event_ids)]

# Martha only edges
martha_edges = edges_df[edges_df['Source'].isin(martha_event_ids) & edges_df['Target'].isin(martha_event_ids)]

# Edges between Origin and Martha
origin_to_martha_edges = edges_df[edges_df['Source'].isin(origin_event_ids) & edges_df['Target'].isin(martha_event_ids)]
martha_to_origin_edges = edges_df[edges_df['Source'].isin(martha_event_ids) & edges_df['Target'].isin(origin_event_ids)]

# Create the graph
G = nx.MultiDiGraph()

# Add Origin nodes
for _, row in origin_events.iterrows():
    G.add_node(str(row['ID']), label=row['Description'], date=row['Date'])

# Add Martha nodes
for _, row in martha_events.iterrows():
    G.add_node(str(row['ID']), label=row['Description'], date=row['Date'])

# Add Origin edges
for _, row in origin_edges.iterrows():
    G.add_edge(str(int(float(row['Source']))), str(int(float(row['Target']))), Type=row['Type'])

# Add Martha edges
for _, row in martha_edges.iterrows():
    G.add_edge(str(int(float(row['Source']))), str(int(float(row['Target']))), Type=row['Type'])

# Add edges between Origin and Martha
for _, row in origin_to_martha_edges.iterrows():
    G.add_edge(str(int(float(row['Source']))), str(int(float(row['Target']))), Type=row['Type'])

for _, row in martha_to_origin_edges.iterrows():
    G.add_edge(str(int(float(row['Source']))), str(int(float(row['Target']))), Type=row['Type'])

# Define sections for Origin
section1 = origin_events[origin_events['Date'].str.contains('1971', na=False)]['ID'].astype(str).tolist()
section2 = origin_events[origin_events['Date'].str.contains('1976', na=False)]['ID'].astype(str).tolist()
section3 = origin_events[origin_events['Date'].str.contains('1986', na=False)]['ID'].astype(str).tolist()

# Define sections for Martha
sections = {}
years = ['1920', '1985', '1986', '2019', '2040', '2052']
for i, year in enumerate(years):
    sections[i] = martha_events[martha_events['Date'].str.contains(year, na=False)]['ID'].astype(str).tolist()

# Circular layout function with spacing
def circular_layout_with_spacing(nodes, center, radius, spacing=0.25, y_spacing=0.65):
    if len(nodes) == 0:
        return {}
    angles = np.linspace(0, 2 * np.pi, len(nodes) + 1)[:-1]
    return {node: (center[0] + (radius + spacing) * np.cos(angle), center[1] + (radius + y_spacing) * np.sin(angle))
            for node, angle in zip(nodes, angles)}

# Assign positions for Origin sections
pos = {}
pos.update(circular_layout_with_spacing(section1, center=(2.5, 0), radius=0.5, y_spacing=1))
pos.update(circular_layout_with_spacing(section2, center=(2.5, 0), radius=1, y_spacing=1))
pos.update(circular_layout_with_spacing(section3, center=(2.5, 0), radius=1.5, y_spacing=1))

# Assign positions for Martha sections
base_radius = 1
radius_step = 0.5  # Increase the radius step to space out the Martha graph
martha_center_x = -5  # Move Martha's graph to the left
for i, nodes in sections.items():
    pos.update(circular_layout_with_spacing(nodes, center=(martha_center_x, 0), radius=base_radius + i * radius_step, y_spacing=1))
#add rip
def add_death_marker(ax, pos, node, img, zoom=0.05, scale_factor=0.0059):
    x, y = pos[node]

    # Calculate node size and determine dynamic offset
    node_size = calculate_node_size(len(wrapped_descriptions[node]))
    node_radius = np.sqrt(node_size) / 2  # Convert to radius

    # Adjust offset dynamically based on node radius
    offset_x = node_radius * scale_factor
    offset_y = node_radius * scale_factor

    # Place the cross image with the adjusted offset
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x + offset_x-0.02, y + offset_y+0.07), frameon=False, xycoords='data', pad=0)
    ax.add_artist(ab)
# Handle nodes not in sections (outliers)
all_nodes = set(G.nodes())
section_nodes = set(section1 + section2 + section3 + [node for sec in sections.values() for node in sec])
nodes_not_in_sections = list(all_nodes - section_nodes)
if nodes_not_in_sections:
    pos.update(circular_layout_with_spacing(nodes_not_in_sections, center=(0, 0), radius=base_radius + len(sections) * radius_step, y_spacing=1))

# Define colors for sections
def generate_color_palette(start_color, end_color, num_colors):
    return [start_color + (end_color - start_color) * i / (num_colors - 1) for i in range(num_colors)]

origin_colors = generate_color_palette(np.array([0.8, 1, 0.8, 0.5]), np.array([0, 0.5, 0, 0.5]), len(section1 + section2 + section3))
martha_colors = generate_color_palette(np.array([0.8, 0.5, 0.8, 0.5]), np.array([0.2, 0, 0.2, 0.5]), len([node for sec in sections.values() for node in sec]))

color_map = {node: origin_colors[i] for i, node in enumerate(section1 + section2 + section3)}
color_map.update({node: martha_colors[i] for i, node in enumerate([node for sec in sections.values() for node in sec])})

# Assign colors to each node
node_colors = [color_map.get(node, 'gray') for node in G.nodes()]

# Function to wrap text into a circular shape with varying width based on node size
def circular_wrap_text(text, max_width=20, min_width=10, node_size=1):
    lines = textwrap.wrap(text, max_width)
    wrapped_lines = []
    for i, line in enumerate(lines):
        width = max_width if len(lines) == 1 else max_width - (max_width - min_width) * (i / (len(lines) - 1))
        width = max(min_width, width * node_size)  # Adjust width based on node size
        wrapped_lines.append(textwrap.fill(line, width=int(width)))
    return "\n".join(wrapped_lines)

# Apply circular wrapping to descriptions with varying node sizes
node_descriptions = {node: G.nodes[node]['label'] for node in G.nodes()}
node_sizes = {node: len(desc) for node, desc in node_descriptions.items()}  # Use description length as a proxy for node size
wrapped_descriptions = {node: circular_wrap_text(desc, max_width=20, min_width=10, node_size=node_sizes[node] / 20) for node, desc in node_descriptions.items()}

# Draw the graph
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(61.44, 12), facecolor='black')  # Increase the height for more y-axis space

# Draw nodes with size based on label length with diminishing increments
def calculate_node_size(length):
    if length <= 10:
        return length * 14
    elif length <= 20:
        return 10 * 14 + (length - 10) * 10
    else:
        return 10 * 14 + 10 * 10 + (length - 20) * 6

# Draw edges with curvature and adjust for node size
edge_colors = []
for source, target, data in G.edges(data=True):  # Ensure edge attributes are included
    edge_type = data.get('Type', '')  # Get edge type, default to empty if not found
    if edge_type == "Succesfull Time Travel":
        edge_colors.append("green")
    elif edge_type == "Failed Time Travel":  # Adjust this if needed
        edge_colors.append("red")
    elif edge_type == "World Swap":
        edge_colors.append("cyan")
    else:
        edge_colors.append("gray")
arc_rad = 0.1
for (edge, color) in zip(G.edges(), edge_colors):
    source, target = edge
    rad = arc_rad if source < target else -arc_rad
    source_pos = pos[source]
    target_pos = pos[target]
    source_size = calculate_node_size(len(wrapped_descriptions[source]))
    target_size = calculate_node_size(len(wrapped_descriptions[target]))
    source_radius = np.sqrt(source_size) / 2
    target_radius = np.sqrt(target_size) / 2

    nx.draw_networkx_edges(
        G, pos, edgelist=[edge], connectionstyle=f'arc3,rad={rad}',
        edge_color=color, alpha=0.5, ax=ax, arrowsize=5,
        min_source_margin=source_radius, min_target_margin=target_radius
    )
node_sizes = [calculate_node_size(len(wrapped_descriptions[node])) for node in G.nodes()]
node_sizes1= {str(node): size for node, size in zip(G.nodes(), node_sizes)}

# Function to add a glow effect by layering multiple semi-transparent circles
def draw_glow_effect(ax, nodes, pos, base_size, layers=5, color='yellow'):
    for i in range(layers, 1, -1):
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes,
            node_size=[(np.sqrt(node_sizes1.get(str(node), 300)) * (1 + i * 0.1))**2 for node in nodes],
            node_color=color,
            alpha=0.1 * (i / layers),  # Decreasing opacity for outer layers
            ax=ax
        )

draw_glow_effect(ax, important_nodes, pos, node_sizes, layers=4, color='yellow')
# Draw normal nodes on top
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)
# Draw labels inside nodes
nx.draw_networkx_labels(G, pos, labels=wrapped_descriptions, font_size=1.5, font_color="white", ax=ax)  # Smaller font size

# Center the graph on the y-axis
y_values = [y for x, y in pos.values()]
y_center = (max(y_values) + min(y_values)) / 2
for node in pos:
    pos[node] = (pos[node][0], pos[node][1] - y_center)

# Save with the correct aspect ratio and higher resolution
for node in death_nodes:
    if node in pos:  # Ensure the node exists in the graph
        add_death_marker(ax, pos, node, cross_img, zoom=0.05)
plt.savefig('graph.png', dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True)  # Higher DPI for higher resolution

# Show the final graph
matplotlib.use('TkAgg')
plt.show()