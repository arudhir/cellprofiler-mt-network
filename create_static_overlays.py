#!/usr/bin/env python3
"""
Create static network overlay images on CellProfiler TIFF.

Simple, reliable approach: load CellProfiler overlay + draw networks on top.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from skimage import io
import os
from mitochondrial_graph_analyzer import MitochondrialGraphAnalyzer


def create_simple_dots_overlay(analyzer, data, overlay_image, output_path):
    """Create overlay with simple red dots for all mitochondria."""
    print("🔴 Creating Simple Dots overlay...")
    
    mito_df = analyzer.preprocess_data(data)
    
    # Create figure with exact image size
    height, width = overlay_image.shape[:2]
    dpi = 100
    figsize = (width/dpi, height/dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Show CellProfiler overlay
    ax.imshow(overlay_image)
    
    # Add red dots for all mitochondria
    ax.scatter(mito_df['Location_Center_X'], mito_df['Location_Center_Y'], 
               c='red', s=20, alpha=0.8, edgecolors='white', linewidths=1)
    
    # Clean formatting
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    print(f"   {len(mito_df)} mitochondria as red dots")


def create_network_overlay(analyzer, data, overlay_image, output_path):
    """Create overlay with network connections colored by cell."""
    print("🌈 Creating Network Connections overlay...")
    
    mito_df = analyzer.preprocess_data(data)
    
    # Create figure with exact image size
    height, width = overlay_image.shape[:2]
    dpi = 100
    figsize = (width/dpi, height/dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Show CellProfiler overlay
    ax.imshow(overlay_image)
    
    # Get unique cells
    cells = mito_df['Parent_Cells'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(cells)))
    
    total_edges = 0
    total_nodes = 0
    
    for i, cell_id in enumerate(cells):
        cell_mito = mito_df[mito_df['Parent_Cells'] == cell_id]
        
        if len(cell_mito) < 2:
            # Single mitochondrion - just show as dot
            if len(cell_mito) == 1:
                ax.scatter(cell_mito['Location_Center_X'], cell_mito['Location_Center_Y'], 
                          c=[colors[i % len(colors)]], s=30, alpha=0.8, 
                          edgecolors='white', linewidths=1)
                total_nodes += 1
            continue
            
        # Build network for this cell
        G = analyzer.build_cell_graph(cell_mito)
        
        if G.number_of_nodes() == 0:
            continue
            
        # Get positions
        positions = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
        
        color = colors[i % len(colors)]
        
        # Draw edges
        if G.number_of_edges() > 0:
            nx.draw_networkx_edges(G, positions, edge_color=color, 
                                 width=2, alpha=0.7, ax=ax)
            total_edges += G.number_of_edges()
        
        # Draw nodes
        nx.draw_networkx_nodes(G, positions, node_size=30, 
                             node_color=color, alpha=0.8, 
                             edgecolors='white', linewidths=1, ax=ax)
        total_nodes += len(positions)
    
    # Clean formatting
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    print(f"   {total_nodes} mitochondria, {total_edges} connections, {len(cells)} cells")


def create_components_overlay(analyzer, data, overlay_image, output_path):
    """Create overlay with unique colors for each connected component."""
    print("🎨 Creating Network Components overlay...")
    
    mito_df = analyzer.preprocess_data(data)
    
    # Create figure with exact image size
    height, width = overlay_image.shape[:2]
    dpi = 100
    figsize = (width/dpi, height/dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Show CellProfiler overlay
    ax.imshow(overlay_image)
    
    # Get unique cells
    cells = mito_df['Parent_Cells'].unique()
    
    # Generate many colors for components
    total_components = 0
    component_colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 20 distinct colors
    color_idx = 0
    
    for cell_id in cells:
        cell_mito = mito_df[mito_df['Parent_Cells'] == cell_id]
        
        if len(cell_mito) == 0:
            continue
            
        # Build network for this cell
        G = analyzer.build_cell_graph(cell_mito)
        
        if G.number_of_nodes() == 0:
            continue
            
        # Get connected components
        components = list(nx.connected_components(G))
        
        for component in components:
            if len(component) == 0:
                continue
                
            # Create subgraph
            subG = G.subgraph(component)
            positions = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in component}
            
            # Assign color
            color = component_colors[color_idx % len(component_colors)]
            color_idx += 1
            
            # Draw edges
            if subG.number_of_edges() > 0:
                nx.draw_networkx_edges(subG, positions, edge_color=color, 
                                     width=2, alpha=0.8, ax=ax)
            
            # Draw nodes
            nx.draw_networkx_nodes(subG, positions, node_size=30, 
                                 node_color=color, alpha=0.8, 
                                 edgecolors='white', linewidths=1, ax=ax)
            
            total_components += 1
    
    # Clean formatting
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    print(f"   {total_components} connected components with unique colors")


def create_connectivity_overlay(analyzer, data, overlay_image, output_path):
    """Create overlay with colors based on connectivity level."""
    print("🌡️ Creating Connectivity-Based overlay...")
    
    mito_df = analyzer.preprocess_data(data)
    
    # Create figure with exact image size
    height, width = overlay_image.shape[:2]
    dpi = 100
    figsize = (width/dpi, height/dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Show CellProfiler overlay
    ax.imshow(overlay_image)
    
    # Get unique cells
    cells = mito_df['Parent_Cells'].unique()
    
    high_conn = 0
    med_conn = 0
    low_conn = 0
    isolated = 0
    
    for cell_id in cells:
        cell_mito = mito_df[mito_df['Parent_Cells'] == cell_id]
        
        if len(cell_mito) == 0:
            continue
            
        # Build network for this cell
        G = analyzer.build_cell_graph(cell_mito)
        
        if G.number_of_nodes() == 0:
            continue
            
        # Calculate average degree
        degrees = [G.degree(node) for node in G.nodes()]
        avg_degree = np.mean(degrees) if degrees else 0
        
        # Assign color based on connectivity
        if avg_degree > 2.0:
            color = 'red'     # High connectivity
            high_conn += 1
        elif avg_degree > 1.0:
            color = 'orange'  # Medium connectivity
            med_conn += 1
        elif avg_degree > 0.1:
            color = 'yellow'  # Low connectivity
            low_conn += 1
        else:
            color = 'gray'    # Isolated
            isolated += 1
        
        # Get positions
        positions = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
        
        # Draw edges
        if G.number_of_edges() > 0:
            nx.draw_networkx_edges(G, positions, edge_color=color, 
                                 width=2, alpha=0.7, ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, positions, node_size=30, 
                             node_color=color, alpha=0.8, 
                             edgecolors='white', linewidths=1, ax=ax)
    
    # Clean formatting
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    print(f"   Red (high): {high_conn}, Orange (med): {med_conn}, Yellow (low): {low_conn}, Gray (isolated): {isolated}")


def main():
    """Create all static overlays."""
    print("🎯 CREATING STATIC NETWORK OVERLAYS")
    print("="*50)
    
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
    
    # Create output directory
    output_dir = "static_overlays"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all overlay types
    overlays = [
        ("simple_dots", "Simple red dots for all mitochondria", create_simple_dots_overlay),
        ("networks", "Networks with connections colored by cell", create_network_overlay),
        ("components", "Connected components with unique colors", create_components_overlay),
        ("connectivity", "Color-coded by connectivity level", create_connectivity_overlay)
    ]
    
    created_files = []
    
    for overlay_name, description, create_func in overlays:
        output_path = os.path.join(output_dir, f"{overlay_name}_overlay.png")
        
        try:
            create_func(analyzer, data, overlay_image, output_path)
            created_files.append((output_path, description))
        except Exception as e:
            print(f"❌ Failed to create {overlay_name}: {e}")
    
    print("\n" + "="*60)
    print("🎉 STATIC OVERLAYS COMPLETE")
    print("="*60)
    
    for file_path, description in created_files:
        print(f"✅ {description}")
        print(f"   📁 {file_path}")
    
    print(f"\n📂 All files in: {output_dir}/")
    print("✨ Perfect coordinate alignment - networks positioned exactly on mitochondria!")
    
    # Create a simple HTML viewer
    create_html_viewer(created_files, output_dir)


def create_html_viewer(created_files, output_dir):
    """Create simple HTML file to view all overlays."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Mitochondrial Network Overlays</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .overlay-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }
        .overlay-item { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .overlay-item h3 { color: #1f77b4; margin-top: 0; }
        .overlay-item img { width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .description { color: #666; margin-bottom: 15px; font-style: italic; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Mitochondrial Network Overlays</h1>
        <p>Networks positioned exactly on CellProfiler segmentation</p>
    </div>
    
    <div class="overlay-grid">
"""
    
    overlay_titles = {
        "simple_dots": "Simple Dots",
        "networks": "Network Connections", 
        "components": "Connected Components",
        "connectivity": "Connectivity Colors"
    }
    
    for file_path, description in created_files:
        filename = os.path.basename(file_path)
        overlay_type = filename.split('_overlay.png')[0]
        title = overlay_titles.get(overlay_type, overlay_type.title())
        
        html_content += f"""
        <div class="overlay-item">
            <h3>{title}</h3>
            <div class="description">{description}</div>
            <img src="{filename}" alt="{title}">
        </div>
        """
    
    html_content += """
    </div>
</body>
</html>
"""
    
    html_path = os.path.join(output_dir, "view_overlays.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"🌐 HTML viewer created: {html_path}")
    print(f"   Open in browser to view all overlays")


if __name__ == "__main__":
    main()