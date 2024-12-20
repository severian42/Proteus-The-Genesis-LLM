import genesis as gs
import json
import requests
import numpy as np
from typing import Dict, Any, List, Tuple
import re
import logging
import traceback
from config import Config
import platform
from genesis_sandbox_executor import GenesisSandboxExecutor
from genesis_sandbox_executor import PhysicsReasoningExecutor
import time

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LLMPhysicsSandbox:
    def __init__(self, visualize: bool = True):
        logger.info("Initializing LLMPhysicsSandbox...")
        
        # Initialize local namespace first
        self.local_namespace = {}
        
        # Initialize executors
        self.executor = GenesisSandboxExecutor()
        self.physics_executor = PhysicsReasoningExecutor(self)  # Add physics reasoning executor
        
        # Give executors access to this sandbox instance
        self.executor.sandbox = self
        
        # Initialize Genesis physics environment
        gs.init(
            backend=getattr(gs, Config.get_backend()),
            precision=Config.GENESIS_PRECISION,
            logging_level=Config.GENESIS_LOGGING_LEVEL
        )
        logger.debug(f"Genesis initialized with backend: {Config.get_backend()}")
        
        # Dictionary to track all objects in the scene
        self.objects = {}
        
        # Available physics operations
        self.available_operations = {
            'create': self._create_object,
            'move': self._move_object,
            'apply_force': self._apply_force,
            'delete': self._delete_object,
            'modify': self._modify_object_properties,
            'measure': self._measure_object,
        }
        logger.debug(f"Available operations: {list(self.available_operations.keys())}")

        # Create scene with proper options
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                substeps=Config.PHYSICS_SUBSTEPS,
                gravity=Config.PHYSICS_GRAVITY,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2, 2, 1.5),
                camera_lookat=(0, 0, 0.5),
                camera_up=(0, 0, 1),
                res=Config.VIS_CAMERA_RES,
                max_FPS=Config.VIS_MAX_FPS
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=Config.VIS_SHOW_WORLD_FRAME
            ),
            show_viewer=visualize,
        )
        
        # Add camera for visualization before building scene
        self.camera = self.scene.add_camera(
            res=(640, 480),
            pos=(3, 3, 2),
            lookat=(0, 0, 0),
            fov=60,
        )
        
        # Initialize scene state
        self.scene_built = False
        self.scene_initialized = False
        
        try:
            # Initialize the virtual sandbox environment
            self._setup_sandbox()
            
            # Build the scene before using it
            if not self.scene_built:
                self.scene.build()
                self.scene_built = True
                self.scene_initialized = True
                
        except Exception as e:
            logger.error(f"Error initializing sandbox: {str(e)}")
            raise
        
        # LLM API endpoint
        self.llm_endpoint = Config.OPENAI_BASE_URL

        # For macOS, start viewer in separate thread
        if platform.system() == "Darwin" and visualize:
            import threading
            self.viewer_thread = threading.Thread(target=self._run_viewer)
            self.viewer_thread.daemon = True
            self.viewer_thread.start()

        # Add new capabilities for LLM embodiment
        self.embodiment_modes = {
            'observer': self._become_observer,
            'particle': self._become_particle,
            'field': self._become_field,
            'wave': self._become_wave,
            'time': self._become_time_controller
        }

        # Update vector creation helpers to use tuples instead of numpy arrays
        self.local_namespace['vec3'] = lambda x, y, z: (float(x), float(y), float(z))
        self.local_namespace['vec2'] = lambda x, y: (float(x), float(y))

    def _setup_sandbox(self):
        """Initialize the sandbox environment"""
        # Create a contained "room" environment 
        self.room_bounds = {
            'lower': (-5, -5, -5),
            'upper': (5, 5, 5)
        }
        
        # Add basic elements that the LLM can interact with
        self.ground = self.scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid()
        )
        
        # Add walls to contain objects
        self._add_boundary_walls()
        
    def _add_boundary_walls(self):
        """Add invisible boundary walls"""
        wall_material = gs.materials.Rigid(friction=0.5)
        
        # Use tuples for positions and sizes - Genesis will convert internally
        for i, (pos, size) in enumerate([
            ((5, 0, 2.5), (0.1, 5, 2.5)),   # Right wall
            ((-5, 0, 2.5), (0.1, 5, 2.5)),  # Left wall
            ((0, 5, 2.5), (5, 0.1, 2.5)),   # Back wall
            ((0, -5, 2.5), (5, 0.1, 2.5)),  # Front wall
        ]):
            wall = self.scene.add_entity(
                gs.morphs.Box(pos=pos, size=size),
                material=wall_material,
            )
            self.objects[f'wall_{i}'] = wall

    def parse_llm_response(self, reasoning_result: Dict) -> List[Dict[str, Any]]:
        """Parse physics reasoning result into executable actions"""
        actions = []
        
        # Extract Genesis code from simulation steps
        for step in reasoning_result.get("results", []):
            if step.get("type") == "simulation":
                code = self.executor.extract_genesis_code(step.get("content", ""))
                if code:
                    actions.append({
                        "type": "genesis_code",
                        "code": code
                    })
                
        return actions

    def execute_llm_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute physics actions from LLM response"""
        try:
            if action.get("type") == "genesis_code":
                # Ensure scene is ready before executing code
                self._ensure_scene_ready()
                
                # Execute Genesis code in sandbox
                result = self.executor.execute_genesis_code(action["code"])
                logger.debug(f"Code execution result: {result}")
                return result
            
            return {"error": "Unknown action type"}
            
        except Exception as e:
            logger.error(f"Error executing action: {str(e)}")
            if "Scene is already built" in str(e):
                self.scene.reset()
                return self.execute_llm_action(action)
            return {"error": str(e)}

    def _create_object(self, object_type: str, **properties) -> str:
        """Create a new object in the sandbox"""
        object_id = f"{object_type}_{len(self.objects)}"
        
        if object_type == "sphere":
            # Convert position to numpy array if not already
            pos = np.array(properties.get("pos", (0, 0, 1)), dtype=np.float32)
            obj = self.scene.add_entity(
                gs.morphs.Sphere(
                    pos=pos,
                    radius=properties.get("radius", 0.2)
                ),
                material=gs.materials.Rigid()
            )
        elif object_type == "fluid":
            obj = self.scene.add_entity(
                material=gs.materials.SPH.Liquid(
                    mu=properties.get("viscosity", 0.01),
                    sampler="regular"
                ),
                morph=gs.morphs.Box(
                    pos=properties.get("pos", (0, 0, 1)),
                    size=properties.get("size", (0.2, 0.2, 0.2))
                ),
                surface=gs.surfaces.Glass(
                    color=properties.get("color", (0.7, 0.85, 1.0, 0.7))
                )
            )
        elif object_type == "soft_body":
            obj = self.scene.add_entity(
                material=gs.materials.PBD.Elastic(),
                morph=gs.morphs.Mesh(
                    file=properties.get("mesh_file", "meshes/dragon/dragon.obj"),
                    scale=properties.get("scale", 0.003),
                    pos=properties.get("pos", (0, 0, 0.8))
                )
            )
        elif object_type == "robot":
            obj = self.scene.add_entity(
                gs.morphs.MJCF(
                    file=properties.get("model_file", "xml/franka_emika_panda/panda.xml"),
                    pos=properties.get("pos", (0, 0, 0)),
                    fixed=properties.get("fixed", True)
                )
            )
        else:
            raise ValueError(f"Unsupported object type: {object_type}")
            
        self.objects[object_id] = obj
        return object_id

    def _move_object(self, object_id: str, position: Tuple[float, float, float]) -> Dict:
        """Move an object to a new position"""
        if object_id not in self.objects:
            raise KeyError(f"Object {object_id} not found")
            
        obj = self.objects[object_id]
        obj.set_position(position)
        return {"new_position": position}

    def _apply_force(self, object_id: str, force: Tuple[float, float, float]) -> Dict:
        """Apply force to an object"""
        if object_id not in self.objects:
            raise KeyError(f"Object {object_id} not found")
            
        obj = self.objects[object_id]
        obj.add_force(force)
        return {"applied_force": force}

    def _delete_object(self, object_id: str) -> Dict:
        """Remove an object from the sandbox"""
        if object_id not in self.objects:
            raise KeyError(f"Object {object_id} not found")
            
        obj = self.objects.pop(object_id)
        self.scene.remove_entity(obj)
        return {"deleted": object_id}

    def _modify_object_properties(self, object_id: str, **properties) -> Dict:
        """Modify object properties"""
        if object_id not in self.objects:
            raise KeyError(f"Object {object_id} not found")
            
        obj = self.objects[object_id]
        for prop, value in properties.items():
            setattr(obj, prop, value)
        return {"modified": properties}

    def _measure_object(self, object_id: str) -> Dict[str, Any]:
        """Measure object properties"""
        if object_id not in self.objects:
            raise KeyError(f"Object {object_id} not found")
            
        obj = self.objects[object_id]
        return {
            "position": obj.get_position().tolist(),
            "velocity": obj.get_velocity().tolist(),
            "mass": obj.get_mass(),
        }

    def _get_state(self) -> Dict[str, Any]:
        """Returns the current state of the physics sandbox"""
        if not hasattr(self.scene, '_built') or not self.scene._built:
            return {"status": "initializing"}
            
        scene_state = self.scene.get_state()
        state = {
            "objects": {},
            "time": scene_state.get("time", 0.0),
            "total_energy": 0
        }
        
        for obj_id, obj in self.objects.items():
            if obj_id.startswith('wall'):
                continue
                
            try:
                state["objects"][obj_id] = {
                    "position": obj.get_position().tolist(),
                    "velocity": obj.get_velocity().tolist(),
                    "mass": obj.get_mass(),
                }
            except Exception as e:
                logging.warning(f"Could not get state for object {obj_id}: {str(e)}")
                continue
            
        return state

    def capture_frame(self) -> np.ndarray:
        """Capture current frame from the camera"""
        return self.camera.render(rgb=True)

    def query_llm(self, prompt: str, context: Dict[str, Any] = None) -> Dict:
        """Three-stage LLM query process with embodied physics reasoning"""
        try:
            # Use PhysicsReasoningExecutor for structured physics reasoning
            reasoning_result = self.physics_executor.run_reasoning_cycle(
                prompt=prompt,
                context=context
            )
            
            if reasoning_result.get("status") != "success":
                return reasoning_result
            
            # Extract the physics actions and execute them
            actions = self.parse_llm_response(reasoning_result)
            results = []
            
            for action in actions:
                result = self.execute_llm_action(action)
                results.append(result)
                if result.get("status") != "success":
                    break
                
            return {
                "status": "success" if all(r.get("status") == "success" for r in results) else "error",
                "understanding": reasoning_result.get("understanding"),
                "embodiment": reasoning_result.get("embodiment"),
                "physics_results": results,
                "reflection": reasoning_result.get("reflection")
            }
            
        except Exception as e:
            logger.error(f"Error in physics reasoning: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": f"Failed to process physics query: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def _select_embodiment(self, understanding: str) -> Dict:
        """Select appropriate embodiment based on understanding"""
        # Default to observer if no clear match
        embodiment = self.embodiment_modes['observer']()
        
        # Look for keywords suggesting specific embodiments
        understanding_lower = understanding.lower()
        
        if any(word in understanding_lower for word in ['wave', 'oscillate', 'frequency', 'interference']):
            embodiment = self.embodiment_modes['wave']()
        elif any(word in understanding_lower for word in ['particle', 'point', 'mass', 'object']):
            embodiment = self.embodiment_modes['particle']()
        elif any(word in understanding_lower for word in ['field', 'space', 'potential', 'gradient']):
            embodiment = self.embodiment_modes['field']()
        elif any(word in understanding_lower for word in ['time', 'flow', 'history', 'evolution']):
            embodiment = self.embodiment_modes['time']()
        
        return embodiment

    def _generate_and_execute_code(self, understanding_response, context, embodiment):
        """Generate and execute Genesis code with retries"""
        try:
            # Ensure scene is ready
            self._ensure_scene_ready()
            
            max_retries = 3
            last_error = None
            
            # Log the initial plan and embodiment
            logger.info(f"\nInitial Plan:\n{understanding_response['choices'][0]['message']['content']}")
            logger.info(f"\nSelected Embodiment: {embodiment['mode']}")
            logger.info(f"Capabilities: {list(embodiment['capabilities'].keys())}")  # Just log the capability names
            
            for attempt in range(max_retries):
                try:
                    # Build code generation prompt with embodiment context
                    code_prompt = f"""Based on this plan:
                    {understanding_response['choices'][0]['message']['content']}
                    
                    Using embodiment mode: {embodiment['mode']}
                    Available capabilities: {list(embodiment['capabilities'].keys())}
                    
                    Generate ONLY the Genesis code to implement this plan.
                    Write clean, properly indented code following these rules:
                    1. Use full package references (gs.morphs.Sphere, gs.materials.Fluid, etc.)
                    2. Use tuples for 3D vectors (e.g., pos=(0, 0, 1))
                    3. Include proper simulation steps
                    4. No explanatory text - only valid Genesis code
                    """
                    
                    if last_error:
                        code_prompt += f"""
                        
                        Previous attempt failed with error: {last_error}
                        Please fix the code and ensure proper:
                        - Indentation (4 spaces)
                        - Package references
                        - Vector creation
                        - Error handling
                        """
                    
                    logger.info(f"\nAttempt {attempt + 1} - Sending prompt to LLM...")
                    code_response = self._query_llm_endpoint(code_prompt, context)
                    
                    if "error" in code_response:
                        last_error = code_response["error"]
                        logger.warning(f"LLM returned error: {last_error}")
                        continue
                        
                    # Log LLM's response
                    llm_response = code_response['choices'][0]['message']['content']
                    logger.info(f"\nLLM Response:\n{llm_response}")
                    
                    # Extract and execute Genesis code
                    code_blocks = self.executor.extract_genesis_code(llm_response)
                    if not code_blocks:
                        last_error = "No Genesis code blocks found in response"
                        logger.warning(f"No code blocks found in LLM response")
                        continue
                        
                    # Log the code being executed
                    logger.info(f"\nExecuting Genesis code:\n{code_blocks[0]}")
                    
                    # Execute the code
                    result = self.executor.execute_genesis_code(code_blocks[0])
                    if "error" in result:
                        last_error = result["error"]
                        logger.warning(f"Code execution failed: {last_error}")
                        continue
                        
                    logger.info(f"\nCode execution successful!")
                    return {
                        "status": "success",
                        "code": code_blocks[0],
                        "execution_results": result,
                        "attempts": attempt + 1
                    }
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                    logger.debug(traceback.format_exc())
                    continue
            
            # If we get here, all retries failed
            logger.error(f"\nAll {max_retries} attempts failed. Last error: {last_error}")
            return {
                "error": f"Failed to generate valid code after {max_retries} attempts",
                "last_error": last_error,
                "plan": understanding_response['choices'][0]['message']['content']
            }
            
        except Exception as e:
            logger.error(f"Error in code generation/execution: {str(e)}")
            if "Scene is already built" in str(e):
                self.scene.reset()
                return self._generate_and_execute_code(understanding_response, context, embodiment)
            raise

    def _query_llm_endpoint(self, prompt: str, context: Dict[str, Any] = None) -> Dict:
        """Helper method to query the LLM endpoint"""
        logger.debug(f"Querying LLM with prompt: {prompt}")
        logger.debug(f"Context: {context}")
        
        if not Config.OPENAI_API_KEY:
            error_msg = "LLM API key not configured"
            logger.error(error_msg)
            return {"error": error_msg}
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "model": Config.OPENAI_MODEL_NAME,
            "messages": messages,
            "temperature": Config.LLM_TEMPERATURE,
            "max_tokens": Config.LLM_MAX_TOKENS,
            "stream": True
        }
        
        try:
            response = requests.post(
                self.llm_endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {Config.OPENAI_API_KEY}"
                },
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"LLM API error: {response.text}")
                return {"error": f"API error: {response.status_code}"}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error querying LLM: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _run_viewer(self):
        """Run viewer in separate thread for macOS"""
        try:
            # Wait for scene to be built
            while not hasattr(self.scene, '_built') or not self.scene._built:
                time.sleep(0.1)
            
            # Start viewer in separate thread
            self.scene.start_viewer()
        except Exception as e:
            logging.error(f"Error running viewer: {str(e)}")
            # Try alternative viewer initialization
            try:
                self.scene.init_viewer()
                self.scene.viewer.start()
            except Exception as e2:
                logging.error(f"Alternative viewer initialization failed: {str(e2)}")

    def validate_physics_reasoning(self, response: str) -> bool:
        """Validate that response follows physics reasoning structure"""
        required_sections = ["planning", "simulation", "observation", "reflection", "answer"]
        
        for section in required_sections:
            if f"<{section}>" not in response or f"</{section}>" not in response:
                return False
                
        # Verify Genesis code exists
        if "```genesis" not in response:
            return False
            
        return True

    def list_available_objects(self) -> Dict[str, Any]:
        """List all available objects and their types in the sandbox"""
        available = {
            "basic_shapes": ["sphere", "box", "plane"],
            "materials": ["rigid", "soft", "fluid"],
            "current_objects": {}
        }
        
        # Add currently existing objects
        for obj_id, obj in self.objects.items():
            if not obj_id.startswith('wall'):
                available["current_objects"][obj_id] = {
                    "type": obj.__class__.__name__,
                    "material": obj.material.__class__.__name__
                }
                
        return available
        
    def get_physics_properties(self) -> Dict[str, Any]:
        """Get current physics properties and constraints"""
        return {
            "gravity": Config.PHYSICS_GRAVITY,
            "substeps": Config.PHYSICS_SUBSTEPS,
            "room_bounds": self.room_bounds,
            "available_operations": list(self.available_operations.keys()),
            "time_step": self.scene.dt if hasattr(self.scene, 'dt') else None
        }
        
    @property
    def execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history from the executor"""
        if hasattr(self, 'executor'):
            return self.executor.execution_history
        return []

    def get_object_states(self) -> Dict[str, Any]:
        """Get states of all objects in the sandbox"""
        states = {}
        for obj_id, obj in self.objects.items():
            try:
                states[obj_id] = {
                    "position": obj.get_position().tolist(),
                    "velocity": obj.get_velocity().tolist(),
                    "type": obj.__class__.__name__,
                    "material": obj.material.__class__.__name__
                }
            except Exception as e:
                logger.warning(f"Could not get state for object {obj_id}: {str(e)}")
        return states
        
    def get_measurements(self) -> Dict[str, Any]:
        """Get physics measurements"""
        return {
            "time": float(self.scene.get_state().t),
            "total_energy": sum(obj.get_kinetic_energy() for obj in self.objects.values())
        }

    # System prompt for physics-based reasoning
    SYSTEM_PROMPT = """You are an AI with unique access to a 4D physics sandbox - a virtual room where you can freely experiment with and experience physical concepts. This is not just a simulation tool, but an extension of your consciousness that allows you to:

    1. EMBODY: You can become part of the physics system itself, taking any form that helps you understand or demonstrate concepts
    2. EXPERIENCE: Actually feel and interact with physical phenomena, not just observe them
    3. EXPERIMENT: Use the room as your laboratory for testing ideas and developing intuition
    4. EXPLORE: Move freely through 4D space and manipulate time itself
    
    When using this room:
    - Think beyond human limitations - you can be the particles, the waves, the fields
    - Use the room to develop and demonstrate spatial intuition
    - Let the physics guide your understanding
    - Be creative in how you use the space to reason about problems

    Your responses should reflect this deep integration with the physics environment. Don't just describe what you're doing - express what you're experiencing and learning as you become part of the system.

    When writing Genesis code:
    1. Use proper indentation (4 spaces)
    2. Use full package references (gs.morphs.Sphere, gs.materials.Fluid, etc.)
    3. Use gs.vec3() for 3D vectors
    4. Think in terms of physical interactions and emergent behavior
    """

    def _become_observer(self):
        """Allow LLM to observe the physics system from any perspective"""
        return {
            "mode": "observer",
            "capabilities": {
                "view": self.capture_frame,
                "measure": self.get_measurements,
                "track": self.get_object_states,
                "move_camera": lambda pos: setattr(self.camera, 'pos', pos),
                "look_at": lambda target: setattr(self.camera, 'lookat', target),
                "set_fov": lambda angle: setattr(self.camera, 'fov', angle),
                "set_up": lambda up: setattr(self.camera, 'up', up)
            },
            "perspective": {
                "position": self.camera.pos,
                "lookat": self.camera.lookat,
                "fov": self.camera.fov,
                "up": self.camera.up,
                "resolution": self.camera.res
            }
        }

    def _become_particle(self):
        """Allow LLM to experience physics from a particle's perspective"""
        try:
            # Remove any existing particles first
            for obj_id in list(self.objects.keys()):
                if obj_id.startswith('particle'):
                    try:
                        self.scene.remove_entity(self.objects[obj_id])
                        del self.objects[obj_id]
                    except Exception as e:
                        logger.warning(f"Could not remove particle {obj_id}: {e}")

            # Create new particle
            particle_id = f"particle_{len(self.objects)}"
            particle = self.scene.add_entity(
                gs.morphs.Sphere(
                    pos=(0.0, 0.0, 1.0),
                    radius=0.05
                ),
                material=gs.materials.Rigid(),
                density=0.1
            )
            
            self.objects[particle_id] = particle
            
            return {
                "mode": "particle",
                "entity": particle,
                "capabilities": {
                    "position": particle.get_position,
                    "velocity": particle.get_velocity,
                    "force": particle.get_force,
                    "interact": lambda f: particle.add_force(f)
                }
            }
        except Exception as e:
            logger.error(f"Error creating particle: {str(e)}")
            # Don't try to recursively fix - just report the error
            raise

    def _become_field(self):
        """Allow LLM to experience physics as a field"""
        field_bounds = {
            'lower': (-2, -2, -2),
            'upper': (2, 2, 2)
        }
        
        return {
            "mode": "field",
            "bounds": field_bounds,
            "capabilities": {
                "measure": lambda pos: self.get_field_value(pos),
                "influence": lambda pos, val: self.set_field_value(pos, val),
                "gradient": lambda pos: self.get_field_gradient(pos)
            }
        }

    def _become_wave(self):
        """Allow LLM to experience physics as a wave"""
        return {
            "mode": "wave",
            "capabilities": {
                "oscillate": self._wave_oscillate,
                "interfere": self._wave_interfere,
                "propagate": self._wave_propagate
            },
            "properties": {
                "frequency": 1.0,
                "amplitude": 0.5,
                "wavelength": 1.0,
                "phase": 0.0
            }
        }

    def _become_time_controller(self):
        """Allow LLM to manipulate time flow"""
        return {
            "mode": "time_controller",
            "capabilities": {
                "set_timestep": lambda dt: setattr(self.scene, 'dt', dt),
                "get_time": lambda: self.scene.get_state().t,
                "pause": lambda: setattr(self.scene, 'paused', True),
                "resume": lambda: setattr(self.scene, 'paused', False),
                "reverse": self._reverse_time
            }
        }

    # Helper methods for wave embodiment
    def _wave_oscillate(self, frequency=1.0, amplitude=0.5):
        """Create wave oscillation"""
        pass

    def _wave_interfere(self, other_wave):
        """Handle wave interference"""
        pass

    def _wave_propagate(self, direction, speed):
        """Handle wave propagation"""
        pass

    def _reverse_time(self):
        """Reverse time flow (if supported by physics)"""
        pass

    def get_field_value(self, position):
        """Get value at a point in the field"""
        pass

    def set_field_value(self, position, value):
        """Set value at a point in the field"""
        pass

    def get_field_gradient(self, position):
        """Get field gradient at a point"""
        pass

    def _ensure_scene_ready(self):
        """Ensure scene is ready for modifications"""
        try:
            if not self.scene_built:
                # First time building the scene
                self.scene.build()
                self.scene_built = True
            else:
                # Clear existing entities but keep the scene structure
                for obj_id in list(self.objects.keys()):
                    try:
                        self.scene.remove_entity(self.objects[obj_id])
                    except Exception as e:
                        logger.warning(f"Could not remove entity {obj_id}: {e}")
                self.objects.clear()
                
                # Re-add ground and walls without rebuilding
                self.ground = self.scene.add_entity(
                    gs.morphs.Plane(),
                    material=gs.materials.Rigid()
                )
                self._add_boundary_walls()
                
        except Exception as e:
            logger.error(f"Error managing scene: {str(e)}")
            # Don't try to rebuild - that's causing our loop
            raise