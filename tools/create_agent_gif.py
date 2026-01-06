#!/usr/bin/env python3
"""
Create an animated GIF showing Multi-Agent System with orchestration and data flow
"""

from PIL import Image, ImageDraw, ImageFont
import imageio
import math
import os

# Configuration
WIDTH, HEIGHT = 800, 600
FPS = 10
DURATION = 3  # seconds
TOTAL_FRAMES = FPS * DURATION

# Colors
BG_COLOR = (240, 245, 250)
ORCHESTRATOR_COLOR = (70, 130, 180)  # Steel blue
AGENT_COLOR = (60, 179, 113)  # Medium sea green
AGENT_ACTIVE_COLOR = (255, 140, 0)  # Dark orange
DATA_FLOW_COLOR = (220, 20, 60)  # Crimson
TASK_COLOR = (138, 43, 226)  # Blue violet
TEXT_COLOR = (50, 50, 50)

def create_frame(frame_num):
    """Create a single frame of the animation"""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Calculate animation progress (0 to 1)
    progress = (frame_num % TOTAL_FRAMES) / TOTAL_FRAMES
    
    # Orchestrator position (center)
    orchestrator_x, orchestrator_y = WIDTH // 2, HEIGHT // 2
    orchestrator_radius = 40
    
    # Agent positions (arranged in a circle around orchestrator)
    num_agents = 5
    agent_radius = 30
    orbit_radius = 200
    
    agents = []
    for i in range(num_agents):
        angle = (2 * math.pi * i / num_agents) - (math.pi / 2)  # Start from top
        x = orchestrator_x + orbit_radius * math.cos(angle)
        y = orchestrator_y + orbit_radius * math.sin(angle)
        agents.append((x, y))
    
    # Draw data flow lines (animated)
    for i, (ax, ay) in enumerate(agents):
        # Calculate flow progress with offset for each agent
        flow_progress = (progress + i * 0.2) % 1.0
        
        # Draw line from orchestrator to agent
        if flow_progress < 0.5:
            # Outgoing data
            t = flow_progress * 2
            start_x = orchestrator_x
            start_y = orchestrator_y
            end_x = ax
            end_y = ay
            current_x = start_x + (end_x - start_x) * t
            current_y = start_y + (end_y - start_y) * t
            
            # Draw line
            draw.line([start_x, start_y, current_x, current_y], 
                     fill=DATA_FLOW_COLOR, width=3)
            # Draw data packet
            packet_size = 8
            draw.ellipse([current_x - packet_size, current_y - packet_size,
                         current_x + packet_size, current_y + packet_size],
                        fill=DATA_FLOW_COLOR)
        else:
            # Incoming data
            t = (flow_progress - 0.5) * 2
            start_x = ax
            start_y = ay
            end_x = orchestrator_x
            end_y = orchestrator_y
            current_x = start_x + (end_x - start_x) * t
            current_y = start_y + (end_y - start_y) * t
            
            # Draw line
            draw.line([start_x, start_y, current_x, current_y], 
                     fill=DATA_FLOW_COLOR, width=3)
            # Draw data packet
            packet_size = 8
            draw.ellipse([current_x - packet_size, current_y - packet_size,
                         current_x + packet_size, current_y + packet_size],
                        fill=DATA_FLOW_COLOR)
    
    # Draw orchestrator
    pulse = 1 + 0.1 * math.sin(progress * 4 * math.pi)
    draw.ellipse([orchestrator_x - orchestrator_radius * pulse,
                 orchestrator_y - orchestrator_radius * pulse,
                 orchestrator_x + orchestrator_radius * pulse,
                 orchestrator_y + orchestrator_radius * pulse],
                fill=ORCHESTRATOR_COLOR, outline=(255, 255, 255), width=3)
    
    # Draw orchestrator label
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
    draw.text((orchestrator_x - 35, orchestrator_y - 5), "ORCHESTRATOR",
             fill=(255, 255, 255), font=font)
    
    # Draw agents
    for i, (ax, ay) in enumerate(agents):
        # Agent activity animation
        agent_active = (progress + i * 0.15) % 1.0 < 0.3
        color = AGENT_ACTIVE_COLOR if agent_active else AGENT_COLOR
        
        # Draw agent circle
        draw.ellipse([ax - agent_radius, ay - agent_radius,
                     ax + agent_radius, ay + agent_radius],
                    fill=color, outline=(255, 255, 255), width=2)
        
        # Draw agent label
        draw.text((ax - 15, ay - 5), f"Agent {i+1}",
                 fill=(255, 255, 255), font=font)
        
        # Draw task indicator when active
        if agent_active:
            task_x = ax + agent_radius + 10
            task_y = ay
            draw.ellipse([task_x - 12, task_y - 12,
                         task_x + 12, task_y + 12],
                        fill=TASK_COLOR, outline=(255, 255, 255), width=2)
            draw.text((task_x - 8, task_y - 5), "TASK",
                     fill=(255, 255, 255), font=font)
    
    # Draw title
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        title_font = ImageFont.load_default()
    draw.text((20, 20), "Multi-Agent System: Orchestration & Data Flow",
             fill=TEXT_COLOR, font=title_font)
    
    # Draw legend
    legend_y = HEIGHT - 100
    draw.text((20, legend_y), "Legend:", fill=TEXT_COLOR, font=font)
    draw.ellipse([20, legend_y + 20, 40, legend_y + 40], fill=ORCHESTRATOR_COLOR)
    draw.text((45, legend_y + 25), "Orchestrator", fill=TEXT_COLOR, font=font)
    draw.ellipse([20, legend_y + 45, 40, legend_y + 65], fill=AGENT_COLOR)
    draw.text((45, legend_y + 50), "Agent (Idle)", fill=TEXT_COLOR, font=font)
    draw.ellipse([200, legend_y + 20, 220, legend_y + 40], fill=AGENT_ACTIVE_COLOR)
    draw.text((225, legend_y + 25), "Agent (Active)", fill=TEXT_COLOR, font=font)
    draw.ellipse([200, legend_y + 45, 220, legend_y + 65], fill=DATA_FLOW_COLOR)
    draw.text((225, legend_y + 50), "Data Flow", fill=TEXT_COLOR, font=font)
    
    return img

def main():
    """Generate the animated GIF"""
    # Get the script directory and navigate to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    print("Creating animated GIF frames...")
    frames = []
    
    for i in range(TOTAL_FRAMES):
        frame = create_frame(i)
        frames.append(frame)
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{TOTAL_FRAMES} frames...")
    
    # Save as GIF (relative to project root)
    output_path = "data/Agent_Output/images/agent-orchestration.gif"
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, duration=1.0/FPS, loop=0)
    print(f"✓ GIF created successfully! ({len(frames)} frames, {DURATION}s duration)")

if __name__ == "__main__":
    main()

