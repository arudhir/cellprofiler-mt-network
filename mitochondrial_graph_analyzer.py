#!/usr/bin/env python3
"""
Mitochondrial Spatial Graph Analysis

Process CellProfiler outputs to build per-cell spatial graphs of mitochondrial networks
and compute biologically meaningful features.

Physical scale: 0.65 microns per pixel
Distance threshold: 10 microns (15.38 pixels)
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import os
from PIL import Image
from skimage import io


class MitochondrialGraphAnalyzer:
    """Analyze mitochondrial spatial networks per cell."""
    
    def __init__(self, pixel_size: float = 0.65, distance_threshold: float = 10.0):
        """
        Initialize analyzer.
        
        Args:
            pixel_size: Microns per pixel
            distance_threshold: Maximum distance in microns for edge creation
        """
        self.pixel_size = pixel_size
        self.distance_threshold = distance_threshold
        self.distance_threshold_pixels = distance_threshold / pixel_size
        
    def load_data(self, mito_path: str, cells_path: str, mito_child_path: Optional[str] = None) -> Dict:
        """
        Load CellProfiler data files.
        
        Args:
            mito_path: Path to MitoObjects.csv
            cells_path: Path to Cells.csv  
            mito_child_path: Optional path to MitoChildObjects.csv
            
        Returns:
            Dictionary containing loaded dataframes
        """
        try:
            mito_df = pd.read_csv(mito_path)
            cells_df = pd.read_csv(cells_path)
            
            # Validate required columns
            required_mito_cols = ['ObjectNumber', 'Location_Center_X', 'Location_Center_Y', 'Parent_Cells']
            missing_cols = [col for col in required_mito_cols if col not in mito_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in MitoObjects.csv: {missing_cols}")
            
            data = {
                'mito': mito_df,
                'cells': cells_df
            }
            
            # Load child objects if available
            if mito_child_path:
                try:
                    child_df = pd.read_csv(mito_child_path)
                    data['mito_child'] = child_df
                except FileNotFoundError:
                    print(f"Warning: MitoChildObjects.csv not found at {mito_child_path}")
                    
            return data
            
        except Exception as e:
            return {"error": f"Error loading data: {str(e)}"}
    
    def preprocess_data(self, data: Dict) -> pd.DataFrame:
        """
        Clean and prepare mitochondria data.
        
        Args:
            data: Dictionary containing dataframes
            
        Returns:
            Cleaned mitochondria dataframe
        """
        mito_df = data['mito'].copy()
        
        # Remove rows with missing coordinates or parent cells
        initial_count = len(mito_df)
        mito_df = mito_df.dropna(subset=['Location_Center_X', 'Location_Center_Y', 'Parent_Cells'])
        
        # Remove invalid coordinates (negative or zero)
        mito_df = mito_df[
            (mito_df['Location_Center_X'] > 0) & 
            (mito_df['Location_Center_Y'] > 0) &
            (mito_df['Parent_Cells'] > 0)
        ]
        
        cleaned_count = len(mito_df)
        print(f"Cleaned data: {initial_count} -> {cleaned_count} mitochondria ({initial_count - cleaned_count} removed)")
        
        return mito_df
    
    def build_cell_graph(self, cell_mito: pd.DataFrame) -> nx.Graph:
        """Build spatial graph for mitochondria within a single cell."""
        return self.build_global_graph(cell_mito)

    def build_global_graph(self, mito_df: pd.DataFrame) -> nx.Graph:
        """Build spatial graph for mitochondria across all cells."""
        G = nx.Graph()

        if len(mito_df) == 0:
            return G

        coords = mito_df[['Location_Center_X', 'Location_Center_Y']].values
        distances = squareform(pdist(coords))

        areas = mito_df.get('AreaShape_Area', pd.Series(np.nan, index=mito_df.index)).fillna(0).values
        radii = np.sqrt(areas / np.pi)

        # Add nodes
        for _, mito in mito_df.iterrows():
            G.add_node(
                mito['ObjectNumber'],
                x=mito['Location_Center_X'],
                y=mito['Location_Center_Y'],
                cell_id=mito['Parent_Cells'],
                area=mito.get('AreaShape_Area', np.nan),
                eccentricity=mito.get('AreaShape_Eccentricity', np.nan),
                solidity=mito.get('AreaShape_Solidity', np.nan),
            )

        mito_ids = mito_df['ObjectNumber'].values
        for i in range(len(mito_ids)):
            for j in range(i + 1, len(mito_ids)):
                dist = distances[i, j]
                touching = dist <= (radii[i] + radii[j])
                if dist <= self.distance_threshold_pixels or touching:
                    G.add_edge(
                        mito_ids[i],
                        mito_ids[j],
                        distance=dist * self.pixel_size,
                    )

        return G
    
    def calculate_graph_metrics(self, G: nx.Graph, cell_id: int) -> Dict:
        """
        Calculate comprehensive graph metrics for a cell.
        
        Args:
            G: NetworkX graph for the cell
            cell_id: Cell identifier
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'cell_id': cell_id,
            'num_mitochondria': G.number_of_nodes(),
            'num_edges': G.number_of_edges()
        }
        
        if G.number_of_nodes() == 0:
            # Empty cell
            metrics.update({
                'avg_degree': 0.0,
                'num_components': 0,
                'diameter': np.nan,
                'avg_shortest_path': np.nan,
                'mean_eccentricity': np.nan,
                'mean_solidity': np.nan,
                'mean_area': np.nan
            })
            return metrics
        
        # Basic metrics
        degrees = [G.degree(n) for n in G.nodes()]
        metrics['avg_degree'] = np.mean(degrees) if degrees else 0.0
        
        # Connected components
        components = list(nx.connected_components(G))
        metrics['num_components'] = len(components)
        
        # Graph connectivity metrics (only for connected graphs)
        if nx.is_connected(G):
            metrics['diameter'] = nx.diameter(G)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            metrics['diameter'] = np.nan
            metrics['avg_shortest_path'] = np.nan
        
        # Node attribute aggregation
        eccentricities = [G.nodes[n].get('eccentricity', np.nan) for n in G.nodes()]
        solidities = [G.nodes[n].get('solidity', np.nan) for n in G.nodes()]
        areas = [G.nodes[n].get('area', np.nan) for n in G.nodes()]
        
        metrics['mean_eccentricity'] = np.nanmean(eccentricities) if eccentricities else np.nan
        metrics['mean_solidity'] = np.nanmean(solidities) if solidities else np.nan
        metrics['mean_area'] = np.nanmean(areas) if areas else np.nan
        
        return metrics
    
    def aggregate_child_features(self, data: Dict, cell_metrics: List[Dict]) -> List[Dict]:
        """
        Aggregate MitoChildObjects features per cell if available.
        
        Args:
            data: Data dictionary containing child objects
            cell_metrics: List of cell metrics to augment
            
        Returns:
            Updated cell metrics with child features
        """
        if 'mito_child' not in data:
            # Add NaN columns for consistency
            for metrics in cell_metrics:
                metrics.update({
                    'mean_child_area': np.nan,
                    'mean_child_eccentricity': np.nan,
                    'mean_child_solidity': np.nan
                })
            return cell_metrics
        
        child_df = data['mito_child']
        mito_df = data['mito']
        
        # Map child objects to cells via parent mitochondria
        child_to_cell = {}
        for _, child in child_df.iterrows():
            parent_mito = child.get('Parent_MitoObjects')
            if pd.notna(parent_mito):
                # Find the cell this mitochondrion belongs to
                parent_cell = mito_df[mito_df['ObjectNumber'] == parent_mito]['Parent_Cells'].values
                if len(parent_cell) > 0:
                    child_to_cell[child['ObjectNumber']] = parent_cell[0]
        
        # Aggregate child features per cell
        cell_child_features = {}
        for child_id, cell_id in child_to_cell.items():
            if cell_id not in cell_child_features:
                cell_child_features[cell_id] = {
                    'areas': [],
                    'eccentricities': [],
                    'solidities': []
                }
            
            child_row = child_df[child_df['ObjectNumber'] == child_id].iloc[0]
            cell_child_features[cell_id]['areas'].append(child_row.get('AreaShape_Area', np.nan))
            cell_child_features[cell_id]['eccentricities'].append(child_row.get('AreaShape_Eccentricity', np.nan))
            cell_child_features[cell_id]['solidities'].append(child_row.get('AreaShape_Solidity', np.nan))
        
        # Add aggregated child features to cell metrics
        for metrics in cell_metrics:
            cell_id = metrics['cell_id']
            if cell_id in cell_child_features:
                features = cell_child_features[cell_id]
                metrics['mean_child_area'] = np.nanmean(features['areas'])
                metrics['mean_child_eccentricity'] = np.nanmean(features['eccentricities']) 
                metrics['mean_child_solidity'] = np.nanmean(features['solidities'])
            else:
                metrics['mean_child_area'] = np.nan
                metrics['mean_child_eccentricity'] = np.nan
                metrics['mean_child_solidity'] = np.nan
        
        return cell_metrics
    
    def analyze_all_cells(self, data: Dict) -> Tuple[pd.DataFrame, str]:
        """
        Build graphs and calculate metrics for all cells.
        
        Args:
            data: Dictionary containing loaded dataframes
            
        Returns:
            Tuple of (results DataFrame, debug summary)
        """
        if 'error' in data:
            return pd.DataFrame(), data['error']
        
        # Preprocess data
        mito_df = self.preprocess_data(data)
        
        if len(mito_df) == 0:
            return pd.DataFrame(), "No valid mitochondria data after preprocessing"
        
        # Group mitochondria by cell
        cell_groups = mito_df.groupby('Parent_Cells')
        
        debug_info = []
        cell_metrics = []
        
        for cell_id, cell_mito in cell_groups:
            # Build spatial graph for this cell
            G = self.build_cell_graph(cell_mito)
            
            # Calculate metrics
            metrics = self.calculate_graph_metrics(G, cell_id)
            cell_metrics.append(metrics)
            
            debug_info.append(f"Cell {cell_id}: {len(cell_mito)} mito, {G.number_of_edges()} edges")
        
        # Add child object features if available
        cell_metrics = self.aggregate_child_features(data, cell_metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(cell_metrics)
        
        # Generate debug summary
        total_cells = len(cell_metrics)
        total_mito = len(mito_df)
        avg_mito_per_cell = total_mito / total_cells if total_cells > 0 else 0
        disconnected_graphs = sum(1 for m in cell_metrics if m['num_components'] > 1)
        
        debug_summary = f"""## Debug Info
- Loaded {total_mito} mitochondria across {total_cells} cells
- Built {total_cells} graphs (mean nodes per graph: {avg_mito_per_cell:.2f})
- {disconnected_graphs} graphs had multiple components
- Distance threshold: {self.distance_threshold} μm ({self.distance_threshold_pixels:.1f} pixels)
"""
        
        return results_df, debug_summary
    
    # ===== VISUALIZATION METHODS =====
    
    def create_population_dashboard(self, results_df: pd.DataFrame, output_dir: str = "visualizations") -> str:
        """
        Create comprehensive population-level visualization dashboard.
        
        Args:
            results_df: DataFrame with cell metrics
            output_dir: Directory to save plots
            
        Returns:
            Path to saved dashboard image
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Network connectivity distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data=results_df, x='num_components', bins=20, ax=ax1)
        ax1.set_title('Network Connectivity Distribution')
        ax1.set_xlabel('Number of Components')
        
        # 2. Mitochondria count vs network complexity
        ax2 = fig.add_subplot(gs[0, 1])
        sns.scatterplot(data=results_df, x='num_mitochondria', y='num_edges', 
                       alpha=0.6, ax=ax2)
        ax2.set_title('Network Complexity vs Mitochondria Count')
        
        # 3. Degree distribution
        ax3 = fig.add_subplot(gs[0, 2])
        sns.histplot(data=results_df, x='avg_degree', bins=30, ax=ax3)
        ax3.set_title('Average Degree Distribution')
        
        # 4. Connected vs disconnected networks
        ax4 = fig.add_subplot(gs[0, 3])
        connected = results_df['num_components'] == 1
        connectivity_counts = [connected.sum(), (~connected).sum()]
        ax4.pie(connectivity_counts, labels=['Connected', 'Disconnected'], 
                autopct='%1.1f%%', startangle=90)
        ax4.set_title('Network Connectivity')
        
        # 5. Morphology correlations
        ax5 = fig.add_subplot(gs[1, 0])
        valid_data = results_df.dropna(subset=['mean_eccentricity', 'avg_degree'])
        sns.scatterplot(data=valid_data, x='mean_eccentricity', y='avg_degree', 
                       alpha=0.6, ax=ax5)
        ax5.set_title('Eccentricity vs Network Degree')
        
        # 6. Size vs connectivity
        ax6 = fig.add_subplot(gs[1, 1])
        valid_data = results_df.dropna(subset=['mean_area', 'num_edges'])
        sns.scatterplot(data=valid_data, x='mean_area', y='num_edges', 
                       alpha=0.6, ax=ax6)
        ax6.set_title('Mitochondrial Size vs Network Edges')
        
        # 7. Network diameter distribution (connected only)
        ax7 = fig.add_subplot(gs[1, 2])
        connected_data = results_df[results_df['num_components'] == 1]
        if len(connected_data) > 0:
            sns.histplot(data=connected_data, x='diameter', bins=20, ax=ax7)
        ax7.set_title('Network Diameter (Connected Networks)')
        
        # 8. Shortest path length distribution
        ax8 = fig.add_subplot(gs[1, 3])
        if len(connected_data) > 0:
            sns.histplot(data=connected_data, x='avg_shortest_path', bins=20, ax=ax8)
        ax8.set_title('Average Shortest Path Length')
        
        # 9. Correlation heatmap
        ax9 = fig.add_subplot(gs[2, :2])
        numeric_cols = ['num_mitochondria', 'num_edges', 'avg_degree', 'num_components',
                       'mean_eccentricity', 'mean_solidity', 'mean_area']
        corr_data = results_df[numeric_cols].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax9)
        ax9.set_title('Feature Correlation Matrix')
        
        # 10. Box plots by connectivity
        ax10 = fig.add_subplot(gs[2, 2])
        results_df['connectivity_type'] = results_df['num_components'].apply(
            lambda x: 'Connected' if x == 1 else 'Disconnected'
        )
        sns.boxplot(data=results_df, x='connectivity_type', y='avg_degree', ax=ax10)
        ax10.set_title('Degree by Connectivity Type')
        
        # 11. Mitochondria count distribution
        ax11 = fig.add_subplot(gs[2, 3])
        sns.histplot(data=results_df, x='num_mitochondria', bins=30, ax=ax11)
        ax11.set_title('Mitochondria per Cell Distribution')
        
        # 12. Summary statistics table
        ax12 = fig.add_subplot(gs[3, :])
        ax12.axis('off')
        
        # Calculate summary stats
        summary_stats = {
            'Total Cells': len(results_df),
            'Mean Mitochondria/Cell': f"{results_df['num_mitochondria'].mean():.2f}",
            'Mean Edges/Cell': f"{results_df['num_edges'].mean():.2f}",
            'Connected Networks': f"{(results_df['num_components'] == 1).sum()} ({(results_df['num_components'] == 1).mean()*100:.1f}%)",
            'Highly Connected (>5 edges)': f"{(results_df['num_edges'] > 5).sum()}",
            'Large Networks (>10 mito)': f"{(results_df['num_mitochondria'] > 10).sum()}"
        }
        
        summary_text = "POPULATION SUMMARY\n" + "\n".join([f"{k}: {v}" for k, v in summary_stats.items()])
        ax12.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle('Mitochondrial Network Population Analysis', fontsize=16, y=0.98)
        
        # Save dashboard
        dashboard_path = os.path.join(output_dir, 'population_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Population dashboard saved to: {dashboard_path}")
        return dashboard_path
    
    def export_networks_to_graphml(self, data: Dict, results_df: pd.DataFrame, 
                                  output_dir: str = "graphml_exports", 
                                  top_n: int = 10) -> List[str]:
        """
        Export interesting cell networks to GraphML for Cytoscape.
        
        Args:
            data: Original data dictionary
            results_df: Analysis results
            output_dir: Directory to save GraphML files
            top_n: Number of top cells to export by different criteria
            
        Returns:
            List of exported file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        exported_files = []
        
        mito_df = self.preprocess_data(data)
        
        # Define selection criteria
        criteria = {
            'high_connectivity': results_df.nlargest(top_n, 'avg_degree'),
            'large_networks': results_df.nlargest(top_n, 'num_mitochondria'),
            'complex_morphology': results_df.nlargest(top_n, 'num_edges'),
            'highly_connected': results_df[results_df['num_components'] == 1].nlargest(top_n, 'diameter')
        }
        
        exported_cells = set()
        
        for criterion, selected_cells in criteria.items():
            for _, cell_row in selected_cells.iterrows():
                cell_id = int(cell_row['cell_id'])
                
                if cell_id in exported_cells:
                    continue
                    
                # Get mitochondria for this cell
                cell_mito = mito_df[mito_df['Parent_Cells'] == cell_id]
                
                if len(cell_mito) == 0:
                    continue
                
                # Build graph
                G = self.build_cell_graph(cell_mito)
                
                # Add additional node attributes for visualization
                for node in G.nodes():
                    node_data = cell_mito[cell_mito['ObjectNumber'] == node].iloc[0]
                    G.nodes[node].update({
                        'label': f"Mito_{node}",
                        'size': float(node_data.get('AreaShape_Area', 10)),
                        'eccentricity_viz': float(node_data.get('AreaShape_Eccentricity', 0.5)),
                        'solidity_viz': float(node_data.get('AreaShape_Solidity', 0.5))
                    })
                
                # Add edge attributes
                for edge in G.edges():
                    G.edges[edge]['weight'] = G.edges[edge]['distance']
                
                # Add graph-level attributes
                G.graph.update({
                    'cell_id': cell_id,
                    'criterion': criterion,
                    'num_mitochondria': len(cell_mito),
                    'avg_degree': cell_row['avg_degree'],
                    'num_components': cell_row['num_components']
                })
                
                # Export to GraphML
                filename = f"cell_{cell_id}_{criterion}.graphml"
                filepath = os.path.join(output_dir, filename)
                nx.write_graphml(G, filepath)
                exported_files.append(filepath)
                exported_cells.add(cell_id)
                
                print(f"Exported {criterion} cell {cell_id}: {len(cell_mito)} mito, {G.number_of_edges()} edges")
        
        # Create summary file
        summary_path = os.path.join(output_dir, "export_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("GraphML Export Summary\n")
            f.write("====================\n\n")
            f.write(f"Total cells exported: {len(exported_cells)}\n")
            f.write(f"Files created: {len(exported_files)}\n\n")
            f.write("Import instructions for Cytoscape:\n")
            f.write("1. Open Cytoscape\n")
            f.write("2. File → Import → Network from File\n")
            f.write("3. Select GraphML files\n")
            f.write("4. Use node attributes (size, eccentricity_viz, solidity_viz) for styling\n")
            f.write("5. Use edge weight for distance-based styling\n\n")
            f.write("Exported files:\n")
            for filepath in exported_files:
                f.write(f"- {os.path.basename(filepath)}\n")
        
        print(f"GraphML export summary saved to: {summary_path}")
        return exported_files
    
    def create_spatial_overlay_plots(self, data: Dict, results_df: pd.DataFrame,
                                   image_dir: str, output_dir: str = "spatial_overlays",
                                   cell_ids: Optional[List[int]] = None) -> List[str]:
        """
        Create spatial network overlays on original microscopy images.
        
        Args:
            data: Original data dictionary
            results_df: Analysis results
            image_dir: Directory containing original images
            output_dir: Directory to save overlay plots
            cell_ids: Specific cell IDs to plot (if None, plot top interesting cells)
            
        Returns:
            List of created plot file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        created_plots = []
        
        mito_df = self.preprocess_data(data)
        
        # Find available images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            image_files.extend(Path(image_dir).glob(ext))
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return created_plots
        
        # Select cells to plot
        if cell_ids is None:
            # Select interesting cells
            top_cells = pd.concat([
                results_df.nlargest(3, 'avg_degree'),
                results_df.nlargest(3, 'num_mitochondria'),
                results_df[results_df['num_components'] == 1].nlargest(2, 'diameter')
            ])['cell_id'].unique()[:8]
        else:
            top_cells = cell_ids
        
        # Load first available image as background
        base_image_path = image_files[0]
        try:
            base_image = io.imread(str(base_image_path))
        except Exception as e:
            print(f"Could not load image {base_image_path}: {e}")
            return created_plots
        
        for cell_id in top_cells:
            try:
                # Get mitochondria for this cell
                cell_mito = mito_df[mito_df['Parent_Cells'] == cell_id]
                
                if len(cell_mito) == 0:
                    continue
                
                # Build graph
                G = self.build_cell_graph(cell_mito)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Show base image with reduced opacity
                ax.imshow(base_image, alpha=0.3, cmap='gray')
                
                # Get positions
                pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
                
                # Draw network
                if G.number_of_edges() > 0:
                    nx.draw_networkx_edges(G, pos, edge_color='yellow', width=2, 
                                         alpha=0.8, ax=ax)
                
                # Draw nodes with size based on area
                node_sizes = [G.nodes[node].get('area', 50) * 3 for node in G.nodes()]
                node_colors = [G.nodes[node].get('eccentricity', 0.5) for node in G.nodes()]
                
                nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                             node_color=node_colors, cmap='viridis',
                                             alpha=0.8, ax=ax)
                
                # Add colorbar for eccentricity
                if nodes:
                    cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
                    cbar.set_label('Eccentricity', rotation=270, labelpad=15)
                
                # Get cell metrics for title
                cell_metrics = results_df[results_df['cell_id'] == cell_id].iloc[0]
                
                title = (f"Cell {cell_id} Mitochondrial Network\n"
                        f"Mito: {cell_metrics['num_mitochondria']}, "
                        f"Edges: {cell_metrics['num_edges']}, "
                        f"Components: {cell_metrics['num_components']}, "
                        f"Avg Degree: {cell_metrics['avg_degree']:.2f}")
                
                ax.set_title(title, fontsize=12, pad=20)
                ax.set_xlabel('X Position (pixels)')
                ax.set_ylabel('Y Position (pixels)')
                
                # Save plot
                plot_path = os.path.join(output_dir, f'cell_{cell_id}_spatial_overlay.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                created_plots.append(plot_path)
                print(f"Created spatial overlay for cell {cell_id}")
                
            except Exception as e:
                print(f"Error creating overlay for cell {cell_id}: {e}")
                continue
        
        return created_plots
    
    def create_cellprofiler_overlay(self, data: Dict, results_df: pd.DataFrame,
                                  overlay_image_path: str, output_dir: str = "cellprofiler_overlays",
                                  min_edges: int = 3) -> str:
        """
        Overlay identified networks on CellProfiler's segmentation overlay image.
        
        Args:
            data: Original data dictionary
            results_df: Analysis results
            overlay_image_path: Path to CellProfiler overlay TIFF
            output_dir: Directory to save overlay plots
            min_edges: Minimum edges to display a network
            
        Returns:
            Path to created overlay image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load CellProfiler overlay image
        try:
            overlay_image = io.imread(overlay_image_path)
            print(f"📸 Loaded overlay image: {overlay_image.shape}")
        except Exception as e:
            print(f"❌ Could not load overlay image {overlay_image_path}: {e}")
            return ""
        
        mito_df = self.preprocess_data(data)
        
        # Filter cells with significant networks
        interesting_cells = results_df[results_df['num_edges'] >= min_edges]['cell_id'].tolist()
        print(f"🔗 Found {len(interesting_cells)} cells with ≥{min_edges} edges")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Display CellProfiler overlay as background
        ax.imshow(overlay_image, alpha=0.8)
        
        # Color scheme for different network types
        colors = plt.cm.Set3(np.linspace(0, 1, len(interesting_cells)))
        
        # Track legend entries
        legend_entries = []
        edge_counts = []
        
        # Overlay networks
        for i, cell_id in enumerate(interesting_cells):
            try:
                # Get mitochondria for this cell
                cell_mito = mito_df[mito_df['Parent_Cells'] == cell_id]
                
                if len(cell_mito) == 0:
                    continue
                
                # Build graph
                G = self.build_cell_graph(cell_mito)
                
                if G.number_of_edges() < min_edges:
                    continue
                
                # Get cell metrics
                cell_metrics = results_df[results_df['cell_id'] == cell_id].iloc[0]
                
                # Get positions
                pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
                
                # Color based on connectivity
                color = colors[i % len(colors)]
                
                # Draw edges
                if G.number_of_edges() > 0:
                    # Edge width based on number of connections
                    edge_widths = []
                    for edge in G.edges():
                        # Thicker edges for shorter distances
                        distance = G.edges[edge]['distance']
                        width = max(1, 4 - (distance / 3))  # Inversely scale with distance
                        edge_widths.append(width)
                    
                    nx.draw_networkx_edges(G, pos, edge_color=color, width=edge_widths,
                                         alpha=0.8, ax=ax)
                
                # Draw nodes
                node_sizes = []
                for node in G.nodes():
                    # Size based on mitochondrial area
                    area = G.nodes[node].get('area', 20)
                    size = max(30, area * 2)  # Scale for visibility
                    node_sizes.append(size)
                
                nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                             node_color=color, alpha=0.9,
                                             edgecolors='black', linewidths=0.5, ax=ax)
                
                # Add to legend
                legend_entries.append(f"Cell {cell_id}")
                edge_counts.append(G.number_of_edges())
                
                print(f"✓ Cell {cell_id}: {len(cell_mito)} mito, {G.number_of_edges()} edges")
                
            except Exception as e:
                print(f"⚠️  Error processing cell {cell_id}: {e}")
                continue
        
        # Customize plot
        ax.set_title(f'Mitochondrial Networks on CellProfiler Segmentation\n'
                    f'{len(interesting_cells)} cells with significant connectivity (≥{min_edges} edges)',
                    fontsize=14, pad=20)
        ax.set_xlabel('X Position (pixels)', fontsize=12)
        ax.set_ylabel('Y Position (pixels)', fontsize=12)
        
        # Create legend with network info
        if legend_entries:
            # Sort by edge count for legend
            sorted_data = sorted(zip(legend_entries, edge_counts), key=lambda x: x[1], reverse=True)
            legend_labels = [f"{label} ({edges} edges)" for label, edges in sorted_data[:10]]  # Top 10
            
            # Create proxy artists for legend
            proxy_artists = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=colors[i % len(colors)], 
                                      markersize=8, alpha=0.9) 
                           for i in range(min(10, len(legend_labels)))]
            
            ax.legend(proxy_artists, legend_labels, 
                     title='Top Connected Networks',
                     loc='upper right', bbox_to_anchor=(1.0, 1.0),
                     framealpha=0.9, fontsize=10)
        
        # Add summary text
        summary_text = (f"Networks: {len(interesting_cells)} cells\n"
                       f"Total mitochondria: {len(mito_df)}\n"
                       f"Distance threshold: {self.distance_threshold} μm")
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Save overlay
        overlay_path = os.path.join(output_dir, 'networks_on_cellprofiler_overlay.png')
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🎨 Network overlay saved to: {overlay_path}")
        return overlay_path
    
    def create_network_summary_overlay(self, data: Dict, results_df: pd.DataFrame,
                                     overlay_image_path: str, output_dir: str = "summary_overlays") -> List[str]:
        """
        Create multiple overlay views showing different network aspects.
        
        Args:
            data: Original data dictionary
            results_df: Analysis results
            overlay_image_path: Path to CellProfiler overlay TIFF
            output_dir: Directory to save overlay plots
            
        Returns:
            List of created overlay image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        created_overlays = []
        
        # Load overlay image
        try:
            overlay_image = io.imread(overlay_image_path)
        except Exception as e:
            print(f"❌ Could not load overlay image: {e}")
            return created_overlays
        
        mito_df = self.preprocess_data(data)
        
        # Create different views
        overlay_configs = [
            {
                'title': 'All Mitochondrial Networks',
                'filename': 'all_networks_overlay.png',
                'filter_func': lambda df: df['num_edges'] > 0,
                'min_edges': 1,
                'description': 'All cells with at least one mitochondrial connection'
            },
            {
                'title': 'Highly Connected Networks',
                'filename': 'highly_connected_overlay.png', 
                'filter_func': lambda df: df['avg_degree'] > 2.0,
                'min_edges': 3,
                'description': 'Cells with high average connectivity (>2.0 degree)'
            },
            {
                'title': 'Large Networks',
                'filename': 'large_networks_overlay.png',
                'filter_func': lambda df: df['num_mitochondria'] > 8,
                'min_edges': 2,
                'description': 'Cells with many mitochondria (>8 organelles)'
            },
            {
                'title': 'Fragmented vs Connected',
                'filename': 'connectivity_comparison_overlay.png',
                'filter_func': lambda df: df['num_mitochondria'] > 3,
                'min_edges': 0,
                'description': 'Comparison of fragmented vs connected networks',
                'special': 'connectivity_comparison'
            }
        ]
        
        for config in overlay_configs:
            try:
                # Filter cells
                filtered_cells = config['filter_func'](results_df)
                
                if len(filtered_cells) == 0:
                    print(f"⚠️  No cells found for {config['title']}")
                    continue
                
                # Create figure
                fig, ax = plt.subplots(figsize=(16, 12))
                ax.imshow(overlay_image, alpha=0.7)
                
                if config.get('special') == 'connectivity_comparison':
                    # Special handling for connectivity comparison
                    connected_cells = filtered_cells[filtered_cells['num_components'] == 1]['cell_id'].tolist()
                    fragmented_cells = filtered_cells[filtered_cells['num_components'] > 1]['cell_id'].tolist()
                    
                    # Plot connected networks in green
                    self._plot_cell_networks(mito_df, connected_cells, ax, color='green', 
                                           label='Connected', alpha=0.8)
                    
                    # Plot fragmented networks in red
                    self._plot_cell_networks(mito_df, fragmented_cells[:len(connected_cells)], ax, 
                                           color='red', label='Fragmented', alpha=0.8)
                    
                    ax.legend(fontsize=12)
                    
                else:
                    # Standard network overlay
                    cell_ids = filtered_cells['cell_id'].tolist()
                    colors = plt.cm.viridis(np.linspace(0, 1, len(cell_ids)))
                    
                    plotted_count = 0
                    for i, cell_id in enumerate(cell_ids):
                        cell_mito = mito_df[mito_df['Parent_Cells'] == cell_id]
                        if len(cell_mito) == 0:
                            continue
                            
                        G = self.build_cell_graph(cell_mito)
                        if G.number_of_edges() < config['min_edges']:
                            continue
                        
                        # Plot network
                        pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
                        
                        if G.number_of_edges() > 0:
                            nx.draw_networkx_edges(G, pos, edge_color=colors[i % len(colors)], 
                                                 width=2, alpha=0.7, ax=ax)
                        
                        node_sizes = [max(30, G.nodes[node].get('area', 20) * 2) for node in G.nodes()]
                        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                             node_color=colors[i % len(colors)], 
                                             alpha=0.8, edgecolors='black', linewidths=0.5, ax=ax)
                        
                        plotted_count += 1
                        if plotted_count >= 50:  # Limit for visibility
                            break
                
                # Customize plot
                ax.set_title(f'{config["title"]}\n{config["description"]}', 
                           fontsize=14, pad=20)
                ax.set_xlabel('X Position (pixels)', fontsize=12)
                ax.set_ylabel('Y Position (pixels)', fontsize=12)
                
                # Save
                overlay_path = os.path.join(output_dir, config['filename'])
                plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                created_overlays.append(overlay_path)
                print(f"✓ Created {config['title']}: {overlay_path}")
                
            except Exception as e:
                print(f"⚠️  Error creating {config['title']}: {e}")
                continue
        
        return created_overlays
    
    def _plot_cell_networks(self, mito_df: pd.DataFrame, cell_ids: List[int], 
                           ax, color: str, label: str, alpha: float = 0.8):
        """Helper function to plot networks for specific cells."""
        for cell_id in cell_ids:
            cell_mito = mito_df[mito_df['Parent_Cells'] == cell_id]
            if len(cell_mito) == 0:
                continue
                
            G = self.build_cell_graph(cell_mito)
            if G.number_of_nodes() == 0:
                continue
            
            pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
            
            # Draw edges
            if G.number_of_edges() > 0:
                nx.draw_networkx_edges(G, pos, edge_color=color, width=2, 
                                     alpha=alpha, ax=ax)
            
            # Draw nodes
            node_sizes = [max(20, G.nodes[node].get('area', 15) * 1.5) for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=color,
                                 alpha=alpha, edgecolors='black', linewidths=0.5, ax=ax)
    
    def create_interactive_dashboard(self, results_df: pd.DataFrame, 
                                   output_dir: str = "interactive") -> str:
        """
        Create interactive Plotly dashboard for web exploration.
        
        Args:
            results_df: Analysis results DataFrame
            output_dir: Directory to save HTML file
            
        Returns:
            Path to HTML dashboard
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Network Connectivity', 'Complexity vs Size', 'Degree Distribution',
                           'Morphology Correlation', 'Diameter Distribution', 'Path Length Distribution',
                           'Feature Correlations', 'Size Distribution', 'Connectivity Summary'),
            specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "heatmap"}, {"type": "histogram"}, {"type": "pie"}]]
        )
        
        # 1. Network connectivity bar chart
        connectivity_counts = results_df['num_components'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=connectivity_counts.index, y=connectivity_counts.values,
                   name='Component Count', showlegend=False),
            row=1, col=1
        )
        
        # 2. Complexity vs size scatter
        fig.add_trace(
            go.Scatter(x=results_df['num_mitochondria'], y=results_df['num_edges'],
                      mode='markers', 
                      text=[f"Cell {id}" for id in results_df['cell_id']],
                      hovertemplate='Cell: %{text}<br>Mitochondria: %{x}<br>Edges: %{y}',
                      name='Cells', showlegend=False),
            row=1, col=2
        )
        
        # 3. Degree distribution
        fig.add_trace(
            go.Histogram(x=results_df['avg_degree'], nbinsx=30, 
                        name='Degree Dist', showlegend=False),
            row=1, col=3
        )
        
        # 4. Morphology correlation
        valid_data = results_df.dropna(subset=['mean_eccentricity', 'avg_degree'])
        fig.add_trace(
            go.Scatter(x=valid_data['mean_eccentricity'], y=valid_data['avg_degree'],
                      mode='markers',
                      text=[f"Cell {id}" for id in valid_data['cell_id']],
                      hovertemplate='Cell: %{text}<br>Eccentricity: %{x:.3f}<br>Degree: %{y:.2f}',
                      name='Morphology', showlegend=False),
            row=2, col=1
        )
        
        # 5. Diameter distribution (connected networks only)
        connected_data = results_df[results_df['num_components'] == 1]
        if len(connected_data) > 0:
            fig.add_trace(
                go.Histogram(x=connected_data['diameter'], nbinsx=20,
                           name='Diameter', showlegend=False),
                row=2, col=2
            )
        
        # 6. Path length distribution
        if len(connected_data) > 0:
            fig.add_trace(
                go.Histogram(x=connected_data['avg_shortest_path'], nbinsx=20,
                           name='Path Length', showlegend=False),
                row=2, col=3
            )
        
        # 7. Correlation heatmap
        numeric_cols = ['num_mitochondria', 'num_edges', 'avg_degree', 'num_components',
                       'mean_eccentricity', 'mean_solidity', 'mean_area']
        corr_matrix = results_df[numeric_cols].corr()
        
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                      colorscale='RdBu', zmid=0, showscale=False),
            row=3, col=1
        )
        
        # 8. Mitochondria count distribution
        fig.add_trace(
            go.Histogram(x=results_df['num_mitochondria'], nbinsx=30,
                        name='Mito Count', showlegend=False),
            row=3, col=2
        )
        
        # 9. Connectivity pie chart
        connected = (results_df['num_components'] == 1).sum()
        disconnected = len(results_df) - connected
        
        fig.add_trace(
            go.Pie(labels=['Connected', 'Disconnected'], values=[connected, disconnected],
                   showlegend=False),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Interactive Mitochondrial Network Analysis Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        # Save interactive plot
        dashboard_path = os.path.join(output_dir, 'interactive_dashboard.html')
        fig.write_html(dashboard_path)
        
        print(f"Interactive dashboard saved to: {dashboard_path}")
        print(f"Open in browser: file://{os.path.abspath(dashboard_path)}")
        
        return dashboard_path
    
    def generate_all_visualizations(self, data: Dict, results_df: pd.DataFrame,
                                  image_dir: Optional[str] = None,
                                  output_base_dir: str = "visualizations") -> Dict[str, str]:
        """
        Generate all visualization types in one command.
        
        Args:
            data: Original data dictionary
            results_df: Analysis results
            image_dir: Directory with original images (optional)
            output_base_dir: Base directory for all outputs
            
        Returns:
            Dictionary mapping visualization type to output path
        """
        print("🎨 Generating comprehensive visualization suite...")
        
        visualization_paths = {}
        
        # 1. Population dashboard
        print("📊 Creating population dashboard...")
        dashboard_path = self.create_population_dashboard(
            results_df, os.path.join(output_base_dir, "population")
        )
        visualization_paths['population_dashboard'] = dashboard_path
        
        # 2. GraphML exports
        print("🔗 Exporting networks to GraphML...")
        graphml_files = self.export_networks_to_graphml(
            data, results_df, os.path.join(output_base_dir, "graphml")
        )
        visualization_paths['graphml_exports'] = graphml_files
        
        # 3. Interactive dashboard
        print("🌐 Creating interactive dashboard...")
        interactive_path = self.create_interactive_dashboard(
            results_df, os.path.join(output_base_dir, "interactive")
        )
        visualization_paths['interactive_dashboard'] = interactive_path
        
        # 4. Spatial overlays (if images available)
        if image_dir and os.path.exists(image_dir):
            print("🖼️ Creating spatial overlay plots...")
            overlay_plots = self.create_spatial_overlay_plots(
                data, results_df, image_dir, 
                os.path.join(output_base_dir, "spatial_overlays")
            )
            visualization_paths['spatial_overlays'] = overlay_plots
        else:
            print("⚠️ Skipping spatial overlays (no image directory provided)")
        
        # 5. CellProfiler overlay (if available)
        cellprofiler_overlay_path = None
        for potential_path in [
            "cp-result/overlay_images/Plate1_AE32_s1_w1_overlay.tiff",
            "overlay_images/Plate1_AE32_s1_w1_overlay.tiff",
            f"{output_base_dir}/../overlay_images/Plate1_AE32_s1_w1_overlay.tiff"
        ]:
            if os.path.exists(potential_path):
                cellprofiler_overlay_path = potential_path
                break
        
        if cellprofiler_overlay_path:
            print("🔬 Creating CellProfiler overlay with networks...")
            cp_overlay = self.create_cellprofiler_overlay(
                data, results_df, cellprofiler_overlay_path,
                os.path.join(output_base_dir, "cellprofiler_overlays")
            )
            if cp_overlay:
                visualization_paths['cellprofiler_overlay'] = cp_overlay
                
                # Create multiple summary overlays
                print("📊 Creating network summary overlays...")
                summary_overlays = self.create_network_summary_overlay(
                    data, results_df, cellprofiler_overlay_path,
                    os.path.join(output_base_dir, "summary_overlays")
                )
                visualization_paths['summary_overlays'] = summary_overlays
        else:
            print("⚠️ CellProfiler overlay image not found - skipping overlay visualizations")
        
        # Create master summary
        summary_path = os.path.join(output_base_dir, "visualization_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Mitochondrial Network Visualization Summary\n")
            f.write("==========================================\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n")
            f.write(f"Analysis of {len(results_df)} cells\n\n")
            
            f.write("Generated Visualizations:\n")
            for viz_type, path in visualization_paths.items():
                if isinstance(path, list):
                    f.write(f"- {viz_type}: {len(path)} files\n")
                else:
                    f.write(f"- {viz_type}: {path}\n")
            
            f.write("\nVisualization Guide:\n")
            f.write("- population_dashboard.png: Overview of all cells and distributions\n")
            f.write("- interactive_dashboard.html: Web-based exploration tool\n")
            f.write("- GraphML files: Import into Cytoscape for network analysis\n")
            f.write("- Spatial overlays: Networks overlaid on microscopy images\n")
        
        visualization_paths['summary'] = summary_path
        
        print(f"✅ All visualizations complete! Summary: {summary_path}")
        return visualization_paths


def main():
    """Main analysis function."""
    # Initialize analyzer
    analyzer = MitochondrialGraphAnalyzer(pixel_size=0.65, distance_threshold=10.0)
    
    # Define file paths
    base_path = "/Users/aru/Development/cellprofiler-mt-network/cp-result"
    mito_path = f"{base_path}/MitoObjects.csv"
    cells_path = f"{base_path}/Cells.csv"
    mito_child_path = f"{base_path}/MitoChildObjects.csv"
    
    # Load data
    print("Loading CellProfiler data...")
    data = analyzer.load_data(mito_path, cells_path, mito_child_path)
    
    if 'error' in data:
        print(f"Error: {data['error']}")
        return
    
    # Analyze all cells
    print("Building spatial graphs and calculating metrics...")
    results_df, debug_summary = analyzer.analyze_all_cells(data)
    
    if len(results_df) == 0:
        print("No results generated - check input data")
        return
    
    # Save results
    output_path = f"{base_path}/mitochondrial_graph_metrics.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Print debug summary
    print(debug_summary)
    
    # Display sample results
    print("\n## Sample Results")
    print(results_df.head().to_string(index=False))
    
    # Summary statistics
    print(f"\n## Summary Statistics")
    print(f"- Total cells analyzed: {len(results_df)}")
    print(f"- Mean mitochondria per cell: {results_df['num_mitochondria'].mean():.2f}")
    print(f"- Mean edges per cell: {results_df['num_edges'].mean():.2f}")
    print(f"- Cells with disconnected graphs: {(results_df['num_components'] > 1).sum()}")
    
    # Generate comprehensive visualizations
    print("\n" + "="*60)
    print("🎨 GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    # Check for image directory
    image_dir = f"{base_path}/../AE32"  # Look for original images
    if not os.path.exists(image_dir):
        image_dir = None
        print("⚠️  Original images not found - will skip spatial overlays")
    
    # Generate all visualizations
    viz_paths = analyzer.generate_all_visualizations(
        data, results_df, image_dir=image_dir, output_base_dir=f"{base_path}/visualizations"
    )
    
    print("\n🎯 VISUALIZATION QUICK ACCESS:")
    print(f"📊 Population Dashboard: {viz_paths.get('population_dashboard', 'N/A')}")
    print(f"🌐 Interactive Dashboard: {viz_paths.get('interactive_dashboard', 'N/A')}")
    print(f"🔗 GraphML Files: {len(viz_paths.get('graphml_exports', []))} networks exported")
    print(f"🖼️  Spatial Overlays: {len(viz_paths.get('spatial_overlays', []))} plots created")
    
    print(f"\n📋 Full summary: {viz_paths.get('summary', 'N/A')}")
    print("\n✅ Analysis and visualization complete!")


if __name__ == "__main__":
    main()