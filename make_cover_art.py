"""
Cover Art Generator
===================

Merges the dynamic Seepage Simulation with the Verification Scorecard
to create a professional "Live Header" for the README.

Layout:
[  3D Seepage Animation  ]  [  Verification Scorecard  ]

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from scaffold_generator import DarkMatterScaffold, ScaffoldParameters
from infall_simulation import SeepageSimulation, SeepageParams

def create_cover_art():
    print("="*60)
    print("GENERATING README COVER ART (Merged GIF)")
    print("="*60)
    
    # 1. Load Scorecard Image
    scorecard_path = 'likelihood_assessment.png'
    try:
        scorecard_img = mpimg.imread(scorecard_path)
        print("1. Loaded Scorecard Image.")
    except Exception as e:
        print(f"Error loading scorecard: {e}")
        return

    # 2. Setup Simulation
    print("2. Initializing Simulation...")
    scaffold = DarkMatterScaffold(ScaffoldParameters(
        grid_size=50, box_size=100.0, # Slightly lower res for speed
        spectral_index=-1.5, smoothing_scale=2.0, random_seed=42
    ))
    scaffold.generate()
    
    params = SeepageParams(
        n_particles=12000, 
        gravity_strength=80.0,
        friction=0.04,
        n_steps=600
    )
    sim = SeepageSimulation(scaffold, params)
    sim.initialize()
    
    # 3. Setup Composite Figure
    # We want side-by-side. 
    # Scorecard is 16:10. Simulation we'll make square.
    # Total aspect ~ 26:10
    fig = plt.figure(figsize=(26, 10), facecolor='black')
    
    # Axes 1: Simulation (Left)
    ax_sim = fig.add_subplot(1, 2, 1, projection='3d', facecolor='black')
    ax_sim.set_xlim(0, 100); ax_sim.set_ylim(0, 100); ax_sim.set_zlim(0, 100)
    ax_sim.axis('off')
    ax_sim.set_title("Dark Scaffold Simulation (Real-Time)", color='cyan', fontsize=20, pad=20)
    
    # Axes 2: Scorecard (Right)
    ax_score = fig.add_subplot(1, 2, 2, facecolor='black')
    ax_score.imshow(scorecard_img)
    ax_score.axis('off')
    # ax_score.set_title("Verification & Viability", color='white', fontsize=20)
    
    # Graphics
    scatter = ax_sim.scatter([], [], [], s=4, c='cyan', alpha=0.8, depthshade=True)
    
    txt = ax_sim.text2D(0.5, 0.05, "Init", transform=ax_sim.transAxes, 
                        ha='center', color='white', fontsize=16)

    print("3. Rendering Frames...")
    
    def update(frame):
        # Physics steps
        steps_per_frame = 4
        for _ in range(steps_per_frame):
            sim.step()
            
        # Draw 3D Scatter
        # Clearing and replotting is robust for 3D rotation if we wanted it
        # But for static cam, updating offsets is tricky in 3D (set_3d_properties not always efficient)
        # Let's use clear/plot approach
        
        ax_sim.clear()
        ax_sim.set_xlim(0, 100); ax_sim.set_ylim(0, 100); ax_sim.set_zlim(0, 100)
        ax_sim.axis('off')
        ax_sim.set_title("Dark Scaffold Simulation (Live)", color='cyan', fontsize=24, pad=20)
        
        # Slowly rotate camera
        ax_sim.view_init(elev=20, azim=frame * 0.5)
        
        p = sim.positions
        # Subsample for render speed
        p_sub = p[::2] 
        ax_sim.scatter(p_sub[:,0], p_sub[:,1], p_sub[:,2], s=5, c='cyan', alpha=0.7)
        
        myr = (frame+1) * steps_per_frame * 2
        ax_sim.text2D(0.5, 0.02, f"Time: {myr} Myr", transform=ax_sim.transAxes, 
                      ha='center', color='white', fontsize=18)
        
        if frame % 10 == 0:
            print(f"   Frame {frame}/150")
            
        return scatter,
        
    # 4. Animate and Save
    # 150 frames @ 20fps = 7.5 seconds loop
    ani = animation.FuncAnimation(fig, update, frames=150, interval=50)
    
    save_path = '/Users/robsimens/Documents/Cosmology/dark-scaffold-theory/README_cover.gif'
    print(f"4. Saving to {save_path}...")
    
    ani.save(save_path, writer='pillow', fps=20, dpi=80) # Lower DPI to keep file size manageable
    print("Done!")

if __name__ == "__main__":
    create_cover_art()
