#!/usr/bin/env python3
"""
Extract and visualize the largest/densest mitochondrial networks.

Identifies networks with the most nodes and highest connectivity, then creates
individual cropped images showing each network in detail.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from skimage import io
import os
from mitochondrial_graph_analyzer import MitochondrialGraphAnalyzer


def analyze_all_networks(analyzer, data):
    """Analyze all networks and rank by size and density."""
    print("🔍 Analyzing all networks...")

    mito_df = analyzer.preprocess_data(data)
    G = analyzer.build_global_graph(mito_df)

    network_stats = []

    components = list(nx.connected_components(G))

    for comp_idx, component in enumerate(components):
        if len(component) == 0:
            continue

        subG = G.subgraph(component)

        num_nodes = len(component)
        num_edges = subG.number_of_edges()
        density = nx.density(subG) if num_nodes > 1 else 0

        degrees = [subG.degree(node) for node in component]
        avg_degree = np.mean(degrees) if degrees else 0

        positions = [(G.nodes[node]['x'], G.nodes[node]['y']) for node in component]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        bbox_width = max(x_coords) - min(x_coords)
        bbox_height = max(y_coords) - min(y_coords)
        bbox_area = bbox_width * bbox_height

        try:
            diameter = nx.diameter(subG) if nx.is_connected(subG) else 0
        except Exception:
            diameter = 0

        cell_ids = {G.nodes[node]['cell_id'] for node in component}

        network_stats.append({
            'cell_id': list(cell_ids)[0] if cell_ids else -1,
            'cells': list(cell_ids),
            'num_cells': len(cell_ids),
            'component': comp_idx,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_degree': avg_degree,
            'diameter': diameter,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'bbox_area': bbox_area,
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords),
            'positions': positions,
        })

    return pd.DataFrame(network_stats)


def extract_network_image(analyzer, data, overlay_image, network_info, output_dir, padding=100):
    """Extract a cropped image showing a specific network."""
    
    # Get network bounds with padding
    x_min = max(0, int(network_info['x_min'] - padding))
    x_max = min(overlay_image.shape[1], int(network_info['x_max'] + padding))
    y_min = max(0, int(network_info['y_min'] - padding))
    y_max = min(overlay_image.shape[0], int(network_info['y_max'] + padding))
    
    # Crop the overlay image
    crop_img = overlay_image[y_min:y_max, x_min:x_max]
    
    mito_df = analyzer.preprocess_data(data)
    G = analyzer.build_global_graph(mito_df)
    components = list(nx.connected_components(G))

    if network_info['component'] >= len(components):
        return None

    target_component = components[network_info['component']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Show cropped background
    ax.imshow(crop_img)
    
    # Draw the target network in bright colors
    positions = {node: (G.nodes[node]['x'] - x_min, G.nodes[node]['y'] - y_min) 
                for node in target_component}
    
    subG = G.subgraph(target_component)
    
    # Draw edges in bright cyan
    if subG.number_of_edges() > 0:
        nx.draw_networkx_edges(subG, positions, edge_color='cyan', 
                             width=4, alpha=0.9, ax=ax)
    
    # Draw nodes in bright yellow with black outlines
    node_sizes = [80 for _ in target_component]
    nx.draw_networkx_nodes(subG, positions, node_size=node_sizes, 
                         node_color='yellow', alpha=0.9, 
                         edgecolors='black', linewidths=2, ax=ax)
    
    # Add node labels
    labels = {node: str(node) for node in target_component}
    nx.draw_networkx_labels(subG, positions, labels, font_size=8, 
                           font_color='black', font_weight='bold', ax=ax)
    
    # Draw other networks in gray for context
    for other_comp_idx, other_component in enumerate(components):
        if other_comp_idx == network_info['component']:
            continue

        other_subG = G.subgraph(other_component)
        other_positions = {node: (G.nodes[node]['x'] - x_min, G.nodes[node]['y'] - y_min)
                          for node in other_component}
        
        # Only draw if in view
        in_view = any(0 <= pos[0] <= (x_max - x_min) and 0 <= pos[1] <= (y_max - y_min) 
                     for pos in other_positions.values())
        
        if not in_view:
            continue
            
        # Draw other networks in gray
        if other_subG.number_of_edges() > 0:
            nx.draw_networkx_edges(other_subG, other_positions, edge_color='gray', 
                                 width=1, alpha=0.5, ax=ax)
        
        nx.draw_networkx_nodes(other_subG, other_positions, node_size=20, 
                             node_color='gray', alpha=0.5, ax=ax)
    
    # Set limits and formatting
    ax.set_xlim(0, x_max - x_min)
    ax.set_ylim(y_max - y_min, 0)
    
    # Add detailed title
    cells_str = ",".join(str(c) for c in network_info.get('cells', [network_info.get('cell_id', '')]))
    title = (
        f"Cells {cells_str}, Component {network_info['component']}\n"
        f"{network_info['num_nodes']} nodes, {network_info['num_edges']} edges\n"
        f"Density: {network_info['density']:.3f}, Avg Degree: {network_info['avg_degree']:.2f}"
    )
    
    ax.set_title(title, fontsize=12, pad=15)
    ax.axis('off')
    
    # Save
    filename = f"network_comp_{network_info['component']}.png"
    output_path = os.path.join(output_dir, filename)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_networks_summary(networks_df, output_dir):
    """Create summary statistics and rankings."""
    
    # Sort by different criteria
    by_size = networks_df.nlargest(20, 'num_nodes')
    by_density = networks_df.nlargest(20, 'density')
    by_edges = networks_df.nlargest(20, 'num_edges')
    by_degree = networks_df.nlargest(20, 'avg_degree')
    
    # Create summary report
    summary_path = os.path.join(output_dir, 'network_rankings.txt')
    
    with open(summary_path, 'w') as f:
        f.write("MITOCHONDRIAL NETWORK RANKINGS\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total networks analyzed: {len(networks_df)}\n")
        f.write(f"Average network size: {networks_df['num_nodes'].mean():.1f} nodes\n")
        f.write(f"Average density: {networks_df['density'].mean():.3f}\n")
        f.write(f"Average degree: {networks_df['avg_degree'].mean():.2f}\n\n")
        
        # Top networks by size
        f.write("TOP 20 NETWORKS BY SIZE (Most Nodes)\n")
        f.write("-" * 40 + "\n")
        for idx, row in by_size.iterrows():
            cells = ",".join(str(c) for c in row['cells'])
            f.write(
                f"Cells {cells}, Comp {row['component']}: "
                f"{row['num_nodes']} nodes, {row['num_edges']} edges, "
                f"density {row['density']:.3f}\n"
            )
        
        f.write(f"\n\nTOP 20 NETWORKS BY DENSITY (Most Connected)\n")
        f.write("-" * 40 + "\n")
        for idx, row in by_density.iterrows():
            cells = ",".join(str(c) for c in row['cells'])
            f.write(
                f"Cells {cells}, Comp {row['component']}: "
                f"density {row['density']:.3f}, {row['num_nodes']} nodes, "
                f"{row['num_edges']} edges\n"
            )
        
        f.write(f"\n\nTOP 20 NETWORKS BY EDGE COUNT\n")
        f.write("-" * 40 + "\n")
        for idx, row in by_edges.iterrows():
            cells = ",".join(str(c) for c in row['cells'])
            f.write(
                f"Cells {cells}, Comp {row['component']}: "
                f"{row['num_edges']} edges, {row['num_nodes']} nodes, "
                f"density {row['density']:.3f}\n"
            )
        
        f.write(f"\n\nTOP 20 NETWORKS BY AVERAGE DEGREE\n")
        f.write("-" * 40 + "\n")
        for idx, row in by_degree.iterrows():
            cells = ",".join(str(c) for c in row['cells'])
            f.write(
                f"Cells {cells}, Comp {row['component']}: "
                f"avg degree {row['avg_degree']:.2f}, {row['num_nodes']} nodes, "
                f"density {row['density']:.3f}\n"
            )
    
    print(f"✅ Network rankings saved: {summary_path}")
    
    return by_size, by_density, by_edges, by_degree


def main():
    """Extract the largest and densest networks."""
    print("🏆 EXTRACTING LARGEST MITOCHONDRIAL NETWORKS")
    print("="*55)
    
    # Load data
    print("📊 Loading data...")
    analyzer = MitochondrialGraphAnalyzer()
    data = analyzer.load_data(
        "cp-result/MitoObjects.csv",
        "cp-result/Cells.csv",
        "cp-result/MitoChildObjects.csv"
    )
    
    if 'error' in data:
        print(f"❌ Error loading data: {data['error']}")
        return
    
    # Load CellProfiler overlay
    overlay_path = "cp-result/overlay_images/Plate1_AE32_s1_w1_overlay.tiff"
    if not os.path.exists(overlay_path):
        print(f"❌ CellProfiler overlay not found: {overlay_path}")
        return
    
    overlay_image = io.imread(overlay_path)
    print(f"✅ Loaded CellProfiler overlay: {overlay_image.shape}")
    
    # Analyze all networks
    networks_df = analyze_all_networks(analyzer, data)
    print(f"✅ Analyzed {len(networks_df)} networks")
    
    # Create output directory
    output_dir = "largest_networks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create rankings
    by_size, by_density, by_edges, by_degree = create_networks_summary(networks_df, output_dir)
    
    # Extract top networks by different criteria
    criteria = [
        (by_size.head(10), "largest", "Largest Networks (Most Nodes)"),
        (by_density.head(10), "densest", "Densest Networks (Highest Density)"),
        (by_edges.head(5), "most_edges", "Most Connected (Most Edges)"),
        (by_degree.head(5), "highest_degree", "Highest Average Degree")
    ]
    
    extracted_files = []
    
    for top_networks, category, description in criteria:
        print(f"\n🔍 Extracting {description}...")
        
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for idx, (_, network_info) in enumerate(top_networks.iterrows()):
            try:
                output_path = extract_network_image(
                    analyzer, data, overlay_image, network_info, category_dir
                )

                if output_path:
                    extracted_files.append((output_path, network_info, description))
                    cells = ",".join(str(c) for c in network_info['cells'])
                    print(
                        f"  ✅ Extracted: Cells {cells}, "
                        f"{network_info['num_nodes']} nodes, "
                        f"density {network_info['density']:.3f}"
                    )

            except Exception as e:
                cells = ",".join(str(c) for c in network_info['cells'])
                print(f"  ❌ Failed to extract Cells {cells}: {e}")
    
    # Create HTML viewer for extracted networks
    create_networks_html_viewer(extracted_files, output_dir)
    
    print(f"\n" + "="*60)
    print("🏆 LARGEST NETWORKS EXTRACTION COMPLETE")
    print("="*60)
    print(f"✅ Analyzed {len(networks_df)} total networks")
    print(f"✅ Extracted {len(extracted_files)} detailed network images")
    print(f"✅ Created rankings and summaries")
    print(f"📂 All files in: {output_dir}/")
    print(f"🌐 View results: open {output_dir}/view_networks.html")


def create_networks_html_viewer(extracted_files, output_dir):
    """Create HTML viewer for all extracted networks."""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Largest Mitochondrial Networks</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .category { margin-bottom: 40px; }
        .category h2 { color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }
        .network-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .network-item { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .network-item img { width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .network-stats { background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.9em; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏆 Largest Mitochondrial Networks</h1>
        <p>Detailed views of the most significant networks</p>
    </div>
"""
    
    # Group files by category
    categories = {}
    for file_path, network_info, description in extracted_files:
        if description not in categories:
            categories[description] = []
        categories[description].append((file_path, network_info))
    
    for category_name, networks in categories.items():
        html_content += f"""
    <div class="category">
        <h2>{category_name}</h2>
        <div class="network-grid">
"""
        
        for file_path, network_info in networks:
            filename = os.path.basename(file_path)
            relative_path = os.path.relpath(file_path, output_dir)
            
            html_content += f"""
            <div class="network-item">
                <img src="{relative_path}" alt="Network Component {network_info['component']}">
                <div class="network-stats">
                    <strong>Cells {','.join(str(c) for c in network_info['cells'])}, Component {network_info['component']}</strong>
                    <div class="stats-grid">
                        <div>Nodes: {network_info['num_nodes']}</div>
                        <div>Edges: {network_info['num_edges']}</div>
                        <div>Density: {network_info['density']:.3f}</div>
                        <div>Avg Degree: {network_info['avg_degree']:.2f}</div>
                    </div>
                </div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    html_path = os.path.join(output_dir, "view_networks.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"🌐 Networks HTML viewer created: {html_path}")


if __name__ == "__main__":
    main()