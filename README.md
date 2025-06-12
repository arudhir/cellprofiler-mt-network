# Mitochondrial Network Analysis

Analyze mitochondrial spatial networks from CellProfiler data and create static overlay visualizations.

## 🚀 Quick Start

```bash
# Create static overlay images
python create_static_overlays.py

# Extract largest/densest networks
python extract_largest_networks.py

# View results
open static_overlays/view_overlays.html
open largest_networks/view_networks.html
```

## 📋 Features

- **CellProfiler Integration**: Load overlay images and segmentation data
- **Network Analysis**: Build spatial graphs of mitochondrial networks  
- **Static Overlays**: Create publication-ready images with networks overlaid on biological structures
- **Multiple Styles**: Four different visualization approaches

## 🎯 Generated Overlays

1. **Simple Dots**: Red dots showing all mitochondria locations
2. **Network Connections**: Networks with connections, colored by cell
3. **Connected Components**: Each connected component uniquely colored  
4. **Connectivity Colors**: Color-coded by connectivity level (red=high, gray=isolated)

## 📁 Key Files

- `create_static_overlays.py` - Create all overlay visualizations
- `extract_largest_networks.py` - Extract and analyze largest/densest networks
- `mitochondrial_graph_analyzer.py` - Core network analysis engine
- `cp-result/` - CellProfiler output data and overlays
- `static_overlays/` - Generated overlay images and HTML viewer
- `largest_networks/` - Detailed views of top networks by size and density

## 📊 Example Results

From your data:
- **2,930 mitochondria** across **973 cells**
- **1,535 network connections** detected
- **1,700 connected components** identified
- **Perfect coordinate alignment** - networks positioned exactly on mitochondria

Perfect coordinate alignment ensures networks appear exactly on mitochondria in the CellProfiler overlay.

# NOTES
Cross-cell mitochondrial networks are now supported. Networks are built across
all cells and connections are created when mitochondria are within the distance
threshold **or when their edges touch**. This allows subnetworks to span
multiple cells when organelles are adjacent.
