<div align="center">
  <h1>Proteus</h1>
  <h3>An LLM-powered physics reasoning engine built on Genesis</h3>
  
  <p><i>Enabling large language models to think, reason, and exist in 4D physical space</i></p>
</div>

---

# What is Proteus?

Proteus is an experimental platform that combines the power of Large Language Models with the Genesis physics engine to create a new form of spatial reasoning and physical intuition. Named after the shape-shifting Greek god, Proteus allows LLMs to transcend their traditional text-based constraints and experience physics concepts directly through embodied reasoning.

## Core Concept

Traditional LLMs excel at processing text but struggle with spatial reasoning and physical intuition. Proteus bridges this gap by providing LLMs with a 4D physics "sandbox" - a virtual room where they can:

- **Embody** physical concepts by becoming particles, waves, fields, or other phenomena
- **Experience** physics directly rather than just describing it
- **Experiment** with ideas in an unconstrained physical space
- **Explore** 4D space-time from novel perspectives

The LLM isn't limited to being an observer - it can become part of the physics system itself, taking any form that helps it understand or demonstrate concepts.

## Key Features

- **Embodied Reasoning**: LLMs can transform into different physical forms (particles, waves, fields) to directly experience concepts
- **4D Physics Sandbox**: Built on the Genesis engine for accurate physics simulation across multiple dimensions
- **Flexible Interaction**: Natural language interface for controlling and experiencing the physics environment
- **Novel Perspectives**: LLMs can reason about physics from "inside" the phenomena themselves
- **Creative Problem-Solving**: Use physical intuition to approach problems in new ways

## Current Status & Limitations

While we've successfully integrated LLMs with the Genesis physics engine, there are some current limitations:

- **Code Execution**: The LLM doesn't always generate valid Genesis code. We're planning to add RAG capabilities with Genesis documentation to improve code generation accuracy.
- **Physics Understanding**: Some complex physics concepts require better prompting or additional context.
- **Performance**: Complex simulations may require optimization.

## Getting Started

```bash
# Install dependencies
pip install genesis-world
pip install -r requirements.txt

# Set up environment
conda create gen-app python=3.10

conda activate gen-app


# Run Proteus
python3 llm_physics_runner.py
```

## ENV Example:
```
# LLM Settings
OPENAI_MODEL_NAME=your-model-name-here
OPENAI_BASE_URL=your-base-url-here
LLM_TEMPERATURE=0.5
LLM_MAX_TOKENS=8192
OPENAI_API_KEY=your-api-key-here  

# Genesis Physics Settings
GENESIS_BACKEND=metal  # Options: metal (Apple Silicon), gpu (NVIDIA), cpu
GENESIS_PRECISION=32
GENESIS_LOGGING_LEVEL=debug
GENESIS_ASSETS_PATH=./assets

# Visualization Settings
VIS_CAMERA_RES=(1280, 720)
VIS_CAMERA_FOV=60
VIS_SHOW_WORLD_FRAME=true
VIS_MAX_FPS=60

# Physics Settings
PHYSICS_GRAVITY=(0, 0, -9.8)
PHYSICS_SUBSTEPS=10
PHYSICS_ROOM_BOUNDS=(-5, -5, -5, 5, 5, 5)
```

## Example Interactions

```python
# Basic Physics Demonstrations
"Show me how gravitational forces work between objects of different masses"
"Create a pendulum system and explain how energy conservation works"
"Demonstrate wave interference using particles in the physics room"

# Creative Physics Problems
"If you were trying to solve the traveling salesman problem, how would you use physics to find a solution?"
"Create a physical representation of a neural network using objects in the room"
"Show me how you would sort numbers using gravity and different mass objects"

# Quantum/Advanced Concepts
"What would it be like to be a wave function? Show me using the physics room"
"Can you become a black hole and demonstrate how spacetime curvature works?"
"Transform the room into a representation of consciousness using physical objects"

# Biological/Chemical Systems
"Create a Rube Goldberg machine that represents the process of photosynthesis"
"Build a physical system that demonstrates how information flows in a computer"
"Show me how evolution works using physical objects and forces"

# Computational Physics
"Using only spheres and gravity, create a clock"
"Design a physical mechanism that can compute basic arithmetic"
"Create a system that can encode and decode binary information using physics"

# Emergent Behavior
"Can you demonstrate how complex patterns emerge from simple rules using particles?"
"Show me how flocking behavior works using multiple objects"
"Create a physical representation of a cellular automaton"

# Higher Dimensions
"How would you represent time as a physical dimension in the room?"
"Show me what a 4D hypercube would look like as it intersects our 3D space"
"Demonstrate how higher dimensional rotation works using 3D projections"

# Musical/Artistic Physics
"Show me what music would look like if it was a physical structure in this space"
"Create a physical representation of color theory using particle interactions"
"Demonstrate how harmony works using wave interference patterns"

# Information Theory
"Show me how entropy looks in a physical system"
"Create a demonstration of Shannon information using particle dynamics"
"Build a physical model of error correction in quantum computing"

# Cognitive Science
"Create a physical model of how memory works in the brain"
"Show me how neural plasticity might look in physical space"
"Demonstrate the concept of attention using particle dynamics"
```

These examples showcase how Proteus can be used to:
- Visualize abstract concepts
- Solve computational problems through physics
- Explore higher-dimensional phenomena
- Model complex systems
- Bridge different domains of knowledge
- Create novel physical metaphors
- Experience quantum concepts classically

## How It Works

1. **Natural Language Interface**: Users interact with Proteus through natural language
2. **LLM Processing**: The LLM interprets the request and plans how to use the physics environment
3. **Embodiment Selection**: The system chooses appropriate physical forms for demonstration
4. **Physics Simulation**: Genesis engine executes the physical demonstration
5. **Experiential Response**: The LLM responds from the perspective of its chosen physical form

## Future Development

- [ ] Add RAG capabilities with Genesis documentation
- [ ] Improve code generation accuracy
- [ ] Expand embodiment options
- [ ] Add more complex physics phenomena
- [ ] Improve visualization capabilities
- [ ] Add collaborative physics reasoning

---

```env
# Genesis Backend Options
GENESIS_BACKEND=metal  # Options: cuda, metal, cpu
GENESIS_PRECISION=32   # Floating point precision
GENESIS_LOGGING_LEVEL=debug
GENESIS_ASSETS_PATH=./genesis/assets

# Physics Room Settings
PHYSICS_GRAVITY=(0, 0, -9.8)  # 3D gravity vector
PHYSICS_SUBSTEPS=10           # Physics simulation substeps
PHYSICS_ROOM_BOUNDS=(-5, -5, -5, 5, 5, 5)  # Room boundaries (x_min, y_min, z_min, x_max, y_max, z_max)

# Visualization Settings
VIS_CAMERA_RES=(1280, 720)  # Render resolution
VIS_CAMERA_FOV=60           # Camera field of view
VIS_SHOW_WORLD_FRAME=true   # Show coordinate frame
VIS_MAX_FPS=60              # Maximum render FPS
```

## LLM Configuration

Configure your LLM settings:

```env
OPENAI_MODEL_NAME=gpt-4-turbo  # Or your preferred model
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TEMPERATURE=0.3
OPENAI_MAX_TOKENS=8192
OPENAI_API_KEY=your-api-key-here
```

## Using the Physics Sandbox

The LLM can interact with the physics environment in several ways:

### Embodiment Modes

The LLM can take on different forms in the physics room:

- **Observer**: Third-person perspective of the physics space
- **Particle**: Experience physics as a point mass
- **Field**: Become a force field or potential
- **Wave**: Experience wave-like phenomena
- **Time Controller**: Manipulate the flow of time

### Example Commands

```python
# Start Proteus
python3 llm_physics_runner.py

# Example interactions:
### Example Interactions

```python
# Basic Physics Demonstrations
"Show me how gravitational forces work between objects of different masses"
"Create a pendulum system and explain how energy conservation works"
"Demonstrate wave interference using particles in the physics room"

# Creative Physics Problems
"If you were trying to solve the traveling salesman problem, how would you use physics to find a solution?"
"Create a physical representation of a neural network using objects in the room"
"Show me how you would sort numbers using gravity and different mass objects"

# Quantum/Advanced Concepts
"What would it be like to be a wave function? Show me using the physics room"
"Can you become a black hole and demonstrate how spacetime curvature works?"
"Transform the room into a representation of consciousness using physical objects"

# Biological/Chemical Systems
"Create a Rube Goldberg machine that represents the process of photosynthesis"
"Build a physical system that demonstrates how information flows in a computer"
"Show me how evolution works using physical objects and forces"

# Computational Physics
"Using only spheres and gravity, create a clock"
"Design a physical mechanism that can compute basic arithmetic"
"Create a system that can encode and decode binary information using physics"

# Emergent Behavior
"Can you demonstrate how complex patterns emerge from simple rules using particles?"
"Show me how flocking behavior works using multiple objects"
"Create a physical representation of a cellular automaton"

# Higher Dimensions
"How would you represent time as a physical dimension in the room?"
"Show me what a 4D hypercube would look like as it intersects our 3D space"
"Demonstrate how higher dimensional rotation works using 3D projections"

# Musical/Artistic Physics
"Show me what music would look like if it was a physical structure in this space"
"Create a physical representation of color theory using particle interactions"
"Demonstrate how harmony works using wave interference patterns"

# Information Theory
"Show me how entropy looks in a physical system"
"Create a demonstration of Shannon information using particle dynamics"
"Build a physical model of error correction in quantum computing"

# Cognitive Science
"Create a physical model of how memory works in the brain"
"Show me how neural plasticity might look in physical space"
"Demonstrate the concept of attention using particle dynamics"
```

These examples showcase how Proteus can be used to:
- Visualize abstract concepts
- Solve computational problems through physics
- Explore higher-dimensional phenomena
- Model complex systems
- Bridge different domains of knowledge
- Create novel physical metaphors
- Experience quantum concepts classically


### Code Generation Guidelines

When the LLM generates Genesis code, it follows these patterns:

```python
# Creating objects
sphere = scene.add_entity(
    gs.morphs.Sphere(pos=(0, 0, 1), radius=0.2),
    material=gs.materials.Rigid()
)

# Applying forces
sphere.add_force((0, 0, -9.81))

# Running simulation steps
for _ in range(100):
    scene.step()
```

## Troubleshooting

Common issues and solutions:

1. **Invalid Genesis Code**: If the LLM generates invalid code, try:
   - Rephrasing your request
   - Using simpler physics concepts
   - Checking Genesis documentation for correct syntax

2. **Visualization Issues**:
   - Ensure proper backend is selected (metal/cuda/cpu)
   - Check camera settings in .env
   - Verify graphics drivers are up to date

3. **Performance Issues**:
   - Reduce physics substeps
   - Lower visualization resolution
   - Simplify physics objects and interactions

## Best Practices

1. **Start Simple**: Begin with basic physics concepts before moving to complex phenomena

2. **Clear Instructions**: Be specific about what physical aspects you want to explore

3. **Use Embodiment**: Let the LLM take on forms that best demonstrate the concept

4. **Monitor Resources**: Complex simulations can be resource-intensive

5. **Save States**: Use the built-in state saving feature for interesting configurations

## Advanced Usage

### Custom Embodiments

You can extend the available embodiment modes by modifying `llm_physics_sandbox.py`:

```python
self.embodiment_modes = {
    'observer': self._become_observer,
    'particle': self._become_particle,
    'field': self._become_field,
    'wave': self._become_wave,
    'time': self._become_time_controller,
    # Add custom modes here
}
```

### Physics Reasoning Pipeline

The LLM follows this reasoning pipeline:

1. **Understanding**: Analyzes the physics concept
2. **Embodiment**: Chooses appropriate physical form
3. **Planning**: Designs demonstration approach
4. **Execution**: Generates and runs Genesis code
5. **Reflection**: Explains experience and insights

This structured approach helps ensure meaningful physics interactions and insights.

## Contributing

We welcome contributions! Whether it's improving the LLM integration, adding new physics capabilities, or enhancing the documentation.

## License

This project is licensed under Apache 2.0. See LICENSE for details.

## Acknowledgments

- Genesis Physics Engine team for the incredible simulation platform
- OpenAI/Anthropic for LLM capabilities
- All contributors to the project

## Citation

```bibtex
@software{Proteus2024,
  author = {Beckett Dillon},
  title = {Proteus: Embodied Physics Reasoning for Large Language Models},
  year = {2024},
  url = {https://github.com/severian42/Proteus-The-Genesis-LLM}
}
```

