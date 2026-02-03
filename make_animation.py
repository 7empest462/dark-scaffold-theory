"""
Seepage Animation Generator
===========================

Generates a high-quality animation of the Dark Scaffold "Seepage" process.
Shows baryonic matter (Cyan) starting from a uniform "Big Bang" distribution
and settling into the pre-existing Dark Matter Scaffold (Orange).

Output: seepage.mp4 (or gif if ffmpeg not found)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters
from infall_simulation import SeepageSimulation, SeepageParams

def create_animation():
    print("="*60)
    print("GENERATING COSMIC SEEPAGE ANIMATION")
    print("="*60)
    
    # 1. Setup Simulation
    print("1. Initializing Universe...")
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=64, box_size=100.0, 
        spectral_index=-1.5, smoothing_scale=2.0, random_seed=42
    ))
    scaffold.generate()
    dm = scaffold.density_field
    
    # 10,000 particles is enough for video (too many slows down rendering)
    params = SeepageParams(
        n_particles=15000, 
        gravity_strength=80.0,
        friction=0.04,
        n_steps=600  # Duration of animation
    )
    sim = SeepageSimulation(scaffold, params)
    sim.initialize()
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(12, 6), facecolor='black')
    
    # Left: 3D Baryon View
    ax3d = fig.add_subplot(1, 2, 1, projection='3d', facecolor='black')
    ax3d.set_xlim(0, 100)
    ax3d.set_ylim(0, 100)
    ax3d.set_zlim(0, 100)
    ax3d.axis('off')
    ax3d.set_title("Baryonic Matter (Visible)", color='cyan', fontsize=14)
    
    # Right: 2D Slice View (Composite)
    ax2d = fig.add_subplot(1, 2, 2, facecolor='black')
    ax2d.axis('off')
    ax2d.set_title("Cross Section: Matter Falling into Scaffold", color='orange', fontsize=14)
    
    # Pre-calculate Scaffold Slice for background
    mid = 32
    dm_slice = dm[:,:,mid]
    
    # Graphics Holders
    scatter = ax3d.scatter([], [], [], s=1.5, c='cyan', alpha=0.6, depthshade=True)
    
    # For 2D view:
    # We will show DM as background heatmap
    im_dm = ax2d.imshow(dm_slice.T, origin='lower', cmap='inferno', alpha=0.6, extent=[0,100,0,100])
    # And particles as overlay scatter
    scatter_2d = ax2d.scatter([], [], s=2, c='cyan', alpha=0.8)
    
    txt = fig.text(0.5, 0.05, "Time: 0 Myr", ha='center', color='white', fontsize=12)
    
    print("2. Running Simulation & Rendering Frames...")
    
    # 3. Animation Update Function
    def update(frame):
        # Run a few physics steps per frame for speed
        steps_per_frame = 3
        for _ in range(steps_per_frame):
            sim.step()
            
        current_step = (frame + 1) * steps_per_frame
        
        # Update 3D Scatter
        # Matplotlib 3D scatter requires specific update method or full redraw
        # Full redraw is safer for 3D scatter
        ax3d.clear()
        ax3d.set_xlim(0, 100); ax3d.set_ylim(0, 100); ax3d.set_zlim(0, 100)
        ax3d.axis('off')
        ax3d.set_title("Baryonic Matter (Visible)", color='cyan', fontsize=14)
        
        # Subsample for speed if needed
        p = sim.positions
        ax3d.scatter(p[:,0], p[:,1], p[:,2], s=1, c='cyan', alpha=0.7, depthshade=False)
        
        # Update 2D Scatter (Project slice)
        # Only show particles within a slice thicknes
        z_mid = 50.0
        thickness = 5.0
        mask = np.abs(p[:,2] - z_mid) < thickness
        p_slice = p[mask]
        
        scatter_2d.set_offsets(p_slice[:,:2])
        
        # Update Text
        # Rough time calibration: say step 1000 = 1 Gyr
        myr = current_step * 2 
        txt.set_text(f"Time: {myr} Myr (z ~ {max(0, 20 - myr/50):.1f})")
        
        if frame % 10 == 0:
            print(f"   Rendering frame {frame}/200...")
            
        return scatter, scatter_2d, txt

    # 4. Create Animation
    # 200 frames * 3 steps = 600 total steps
    ani = animation.FuncAnimation(fig, update, frames=200, interval=20, blit=False)
    
    # 5. Save
    print("3. Saving Video (this may take a minute)...")
    try:
        # Try MP4 first (requires ffmpeg)
        # ani.save('seepage_video.mp4', writer='ffmpeg', fps=30, dpi=150)
        # To avoid ffmpeg dependency issues, let's use GIF which is universally supported by Pillow
        ani.save('/Users/robsimens/Documents/Cosmology/dark-scaffold-theory/seepage.gif', writer='pillow', fps=30, dpi=120)
        print("   Success! Saved to seepage.gif")
    except Exception as e:
        print(f"   Error saving video: {e}")
        
    print("="*60)

if __name__ == "__main__":
    create_animation()
