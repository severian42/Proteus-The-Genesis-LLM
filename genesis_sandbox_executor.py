import re
from typing import Tuple, List, Dict, Any
import tempfile
import json
import logging
import traceback
from pathlib import Path
import genesis as gs
from fastapi import HTTPException
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

class GenesisSandboxExecutor:
    """Safe execution environment for Genesis physics code"""
    
    ALLOWED_IMPORTS = {
        'genesis': gs,
        'numpy': 'np',
        'math': 'math'
    }
    
    ALLOWED_OPERATIONS = {
        'add_entity',
        'remove_entity',
        'get_state',
        'step',
        'build',
        'add_camera',
        'render'
    }

    def __init__(self):
        self.local_namespace = {}
        self.local_namespace['gs'] = gs
        self.local_namespace['scene'] = None
        self.local_namespace['math'] = __import__('math')
        self.local_namespace['np'] = np
        
        for module, alias in self.ALLOWED_IMPORTS.items():
            if isinstance(alias, str):
                exec(f"import {module} as {alias}", self.local_namespace)
            else:
                self.local_namespace[module] = alias
        
        self.execution_history = []
        self.max_executions_per_minute = 60
        self.max_code_length = 10000
        
        # Add Genesis-specific helpers using proper types
        self.local_namespace['get_time'] = lambda: self.local_namespace['scene'].get_state().t
        # Use tuple for vectors
        self.local_namespace['vec3'] = lambda x, y, z: np.array([x, y, z], dtype=np.float32)
        self.local_namespace['vec2'] = lambda x, y: np.array([x, y], dtype=np.float32)
        
        # Add helper functions
        self.local_namespace['create_fluid_particles'] = self._create_fluid_particles
        self.local_namespace['create_spring_mass'] = self._create_spring_mass
        self.local_namespace['create_wave_sources'] = self._create_wave_sources
        
    def _check_rate_limit(self):
        """Check if we've exceeded rate limits"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Clean old history
        self.execution_history = [t for t in self.execution_history if t > cutoff]
        
        if len(self.execution_history) >= self.max_executions_per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        self.execution_history.append(now)

    def extract_genesis_code(self, text: str) -> List[str]:
        """Extract Genesis code blocks from text."""
        pattern = r'```genesis\s*(.*?)\s*```'
        return re.findall(pattern, text, re.DOTALL)

    def validate_code(self, code: str) -> bool:
        """Validate Genesis code for safety."""
        # Check for forbidden imports and dangerous operations
        dangerous_patterns = [
            r'import\s+(?!genesis|numpy|math)',  # Only allow specific imports
            r'open\(',  # No file operations
            r'os\.',   # No OS operations
            r'system\(',  # No system calls
            r'eval\(',  # No eval
            r'exec\(',  # No exec
            r'subprocess', # No subprocess
            r'__import__', # No dynamic imports
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code.lower()):
                logger.warning(f"Dangerous pattern found: {pattern}")
                return False
            
        # Allow common Genesis operations with more permissive patterns
        allowed_patterns = [
            # Scene operations
            r'scene\.(add_entity|step|apply_force|get_time|get_state)',
            
            # Genesis objects and methods
            r'gs\.(morphs|materials|vec3|vec2|pi|sin|cos|options)',
            r'gs\.morphs\.(Sphere|Box|Plane|Mesh|MJCF)',
            r'gs\.materials\.(Rigid|Fluid|SPH|MPM|PBD)',
            r'gs\.surfaces\.(Default|Glass|Metal)',
            
            # Math operations
            r'math\.(sin|cos|pi|sqrt|pow)',
            r'numpy\.(sin|cos|pi|array|zeros|ones)',
            
            # Control structures
            r'for\s+.*\s+in\s+range',
            r'if\s+.*:',
            r'else:',
            r'elif\s+.*:',
            r'while\s+.*:',
            
            # Variable assignments and property access
            r'\w+\s*=\s*.*',  # More permissive assignment pattern
            r'\w+\.(pos|radius|material|density|size|scale|color|surface|fixed)',
            r'\w+\s*:\s*.*',  # Allow dict/parameter specifications
            
            # Function calls and parameters
            r'\w+\([^)]*\)',  # General function calls
            r'vec[23]\([^)]*\)',  # Vector constructors
            
            # Comments
            r'#.*$',
        ]
        
        # Split code into lines and check each non-empty line
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        for line in lines:
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Check if line matches any allowed pattern
            if not any(re.search(pattern, line) for pattern in allowed_patterns):
                # More permissive fallback check for basic operations
                if re.match(r'^[\w\s\(\)\[\]\{\},\.:=\+\-\*/]*$', line):
                    continue
                logger.debug(f"Line failed validation: {line}")  # Changed to debug level
                return False
        
        return True

    def preprocess_code(self, code: str) -> str:
        """Fix common LLM code generation issues"""
        def fix_indentation(code_str: str) -> str:
            """Fix indentation while preserving code structure"""
            lines = code_str.split('\n')
            fixed_lines = []
            indent_level = 0
            in_parentheses = 0  # Track nested parentheses
            
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                
                # Track parentheses nesting
                in_parentheses += stripped.count('(') - stripped.count(')')
                
                # Handle control structures
                if any(stripped.startswith(keyword) for keyword in ['for', 'if', 'while', 'def', 'class']):
                    fixed_lines.append('    ' * indent_level + stripped)
                    if stripped.endswith(':'):
                        indent_level += 1
                elif stripped.startswith(('elif', 'else:', 'except:', 'finally:')):
                    indent_level = max(0, indent_level - 1)
                    fixed_lines.append('    ' * indent_level + stripped)
                    if stripped.endswith(':'):
                        indent_level += 1
                else:
                    # Handle multi-line statements
                    if in_parentheses > 0:
                        # Inside parentheses - maintain indentation
                        fixed_lines.append('    ' * (indent_level + 1) + stripped)
                    else:
                        # Normal line
                        fixed_lines.append('    ' * indent_level + stripped)
                    
                    # Adjust indentation after block ends
                    if stripped.startswith(('return', 'break', 'continue', 'pass')):
                        indent_level = max(0, indent_level - 1)
                
            return '\n'.join(fixed_lines)
        
        # Remove scene creation if present
        code = re.sub(r'scene\s*=\s*Scene\(\)', '', code)
        code = re.sub(r'scene\s*=\s*gs\.Scene\(\)', '', code)
        
        # Fix morphs references
        code = re.sub(r'(?<!gs\.morphs\.)Sphere\(', 'gs.morphs.Sphere(', code)
        code = re.sub(r'(?<!gs\.morphs\.)Box\(', 'gs.morphs.Box(', code)
        code = re.sub(r'(?<!gs\.morphs\.)Plane\(', 'gs.morphs.Plane(', code)
        
        # Fix materials references
        code = re.sub(r'(?<!gs\.materials\.)Fluid\(', 'gs.materials.Fluid(', code)
        code = re.sub(r'(?<!gs\.materials\.)Rigid\(', 'gs.materials.Rigid(', code)
        code = re.sub(r'(?<!gs\.materials\.SPH\.)Liquid\(', 'gs.materials.SPH.Liquid(', code)
        
        # Fix vector creation
        code = re.sub(r'gs\.vec3\(', 'vec3(', code)
        code = re.sub(r'gs\.vec2\(', 'vec2(', code)
        code = re.sub(r'(?<!vec)3\(([^)]+)\)', r'vec3(\1)', code)
        
        # Apply indentation fixing
        code = fix_indentation(code)
        
        return code

    def execute_genesis_code(self, code: str) -> Dict[str, Any]:
        """Execute Genesis code in a sandboxed environment with retries."""
        max_retries = 3
        attempt = 0
        last_error = None

        while attempt < max_retries:
            try:
                attempt += 1
                logger.info(f"Attempt {attempt} of {max_retries}")

                # Preprocess code first
                processed_code = self.preprocess_code(code)
                logger.debug(f"Preprocessed code:\n{processed_code}")
                
                # Validate code with better error reporting
                if not self.validate_code(processed_code):
                    last_error = "Code validation failed - Invalid syntax or unsafe operations"
                    logger.warning(f"Attempt {attempt}: {last_error}")
                    logger.debug(f"Failed code:\n{processed_code}")
                    if attempt == max_retries:
                        return {"error": last_error, "code": processed_code}
                    continue

                if len(processed_code) > self.max_code_length:
                    return {"error": "Code exceeds maximum length"}
                
                self._check_rate_limit()
                
                # Get current scene from LLMPhysicsSandbox
                if hasattr(self, 'sandbox'):
                    self.local_namespace['scene'] = self.sandbox.scene
                
                # Execute the code with proper error handling
                try:
                    # Properly indent the code for the try block
                    indented_code = '\n'.join(f"    {line}" for line in processed_code.split('\n') if line.strip())
                    wrapped_code = f"""
try:
{indented_code}
except Exception as e:
    print(f"Runtime error: {{str(e)}}")
    raise
"""
                    logger.debug(f"Wrapped code:\n{wrapped_code}")
                    exec(wrapped_code, self.local_namespace)
                    
                    # Get scene state after execution
                    state = self.local_namespace['scene'].get_state() if self.local_namespace['scene'] else {}
                    
                    return {
                        "status": "success",
                        "state": state,
                        "output": "Code executed successfully",
                        "attempts": attempt
                    }
                    
                except (IndentationError, SyntaxError) as e:
                    last_error = f"Code formatting error: {str(e)}"
                    logger.warning(f"Attempt {attempt}: {last_error}")
                    # Try to fix common formatting issues
                    code = self._fix_common_errors(code, str(e))
                    continue
                    
                except Exception as e:
                    last_error = f"Runtime error: {str(e)}"
                    logger.warning(f"Attempt {attempt}: {last_error}")
                    if attempt == max_retries:
                        return {
                            "status": "error",
                            "error": last_error,
                            "traceback": traceback.format_exc()
                        }
                    continue
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt}: {last_error}")
                logger.debug(f"Failed code:\n{code}")
                if attempt == max_retries:
                    return {
                        "error": str(e),
                        "code": code,
                        "traceback": traceback.format_exc()
                    }
                continue
        
        # If we get here, all retries failed
        return {
            "error": f"Failed to execute code after {max_retries} attempts",
            "last_error": last_error,
            "code": code
        }

    def _fix_common_errors(self, code: str, error_msg: str) -> str:
        """Attempt to fix common code errors based on error message"""
        # Fix indentation errors
        if "IndentationError" in error_msg:
            lines = code.split('\n')
            fixed_lines = []
            current_indent = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                    
                # Adjust indent for control structures
                if any(stripped.startswith(word) for word in ['if', 'for', 'while', 'def', 'class']):
                    fixed_lines.append('    ' * current_indent + stripped)
                    if stripped.endswith(':'):
                        current_indent += 1
                elif stripped.startswith(('elif', 'else:', 'except:', 'finally:')):
                    current_indent = max(0, current_indent - 1)
                    fixed_lines.append('    ' * current_indent + stripped)
                    if stripped.endswith(':'):
                        current_indent += 1
                else:
                    fixed_lines.append('    ' * current_indent + stripped)
                    
            return '\n'.join(fixed_lines)
            
        # Fix syntax errors
        if "SyntaxError" in error_msg:
            # Fix missing parentheses
            if "EOF while scanning" in error_msg:
                return code + '\n)'
            # Fix missing colons
            if "expected ':'" in error_msg:
                return code.replace('\n', ':\n', 1)
                
        return code

    def enforce_genesis_usage(self, response: Dict) -> Dict:
        """Ensure response includes Genesis physics code"""
        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if '```genesis' not in content:
            logger.warning("Response missing Genesis code block - enforcing usage")
            # Add Genesis code section
            physics_code = self._generate_default_physics_code()
            content = content.replace(
                "</planning>",
                f"</planning>\n\n<simulation>\n```genesis\n{physics_code}\n```\n</simulation>"
            )
            response['choices'][0]['message']['content'] = content
            
        return response
        
    def _generate_default_physics_code(self) -> str:
        """Generate default physics code based on context"""
        return """
        # Create basic physics demonstration
        sphere = scene.add_entity(
            gs.morphs.Sphere(pos=(0, 0, 1), radius=0.2),
            material=gs.materials.Rigid(),
            density=1.0
        )
        
        # Run simulation
        for _ in range(50):
            scene.step()
        """

    def _create_fluid_particles(self, bounds, spacing, properties):
        """Helper to create fluid particle grid"""
        particles = []
        for x in range(bounds[0], bounds[1], spacing):
            for y in range(bounds[2], bounds[3], spacing):
                for z in range(bounds[4], bounds[5], spacing):
                    pos = np.array([x, y, z], dtype=np.float32)
                    particle = self.local_namespace['scene'].add_entity(
                        gs.morphs.Sphere(
                            pos=pos,
                            radius=properties.get('radius', 0.05)
                        ),
                        material=gs.materials.Fluid(
                            mu=properties.get('mu', 0.01),
                            bulk_modulus=properties.get('bulk_modulus', 1.0)
                        ),
                        density=properties.get('density', 1.0)
                    )
                    particles.append(particle)
        return particles

    def _create_spring_mass(self, anchor_pos, mass_pos, properties):
        """Helper to create spring-mass system"""
        # Create anchor
        anchor = self.local_namespace['scene'].add_entity(
            gs.morphs.Sphere(
                pos=anchor_pos,
                radius=properties.get('anchor_radius', 0.1)
            ),
            material=gs.materials.Rigid(),
            density=properties.get('anchor_density', 1.0),
            fixed=True
        )
        
        # Create mass
        mass = self.local_namespace['scene'].add_entity(
            gs.morphs.Sphere(
                pos=mass_pos,
                radius=properties.get('mass_radius', 0.2)
            ),
            material=gs.materials.Rigid(),
            density=properties.get('mass_density', 2.0)
        )
        
        # Add spring constraint
        spring = self.local_namespace['scene'].add_constraint(
            "spring",
            anchor, mass,
            stiffness=properties.get('stiffness', 100.0),
            damping=properties.get('damping', 1.0),
            rest_length=properties.get('rest_length', 1.0)
        )
        
        return anchor, mass, spring

    def _create_wave_sources(self, positions, properties):
        """Helper to create wave sources"""
        sources = []
        for pos in positions:
            source = self.local_namespace['scene'].add_entity(
                gs.morphs.Sphere(
                    pos=pos,
                    radius=properties.get('radius', 0.1)
                ),
                material=gs.materials.Fluid(
                    mu=properties.get('mu', 0.1),
                    bulk_modulus=properties.get('bulk_modulus', 1.0)
                ),
                density=properties.get('density', 1.0)
            )
            sources.append(source)
        return sources

    def _create_observation_plane(self, pos, size, properties):
        """Helper to create observation plane"""
        return self.local_namespace['scene'].add_entity(
            gs.morphs.Plane(
                pos=pos,
                size=size
            ),
            material=gs.materials.Soft(
                mu=properties.get('mu', 0.05)
            ),
            density=properties.get('density', 0.1)
        )

    def debug_code_validation(self, code: str) -> None:
        """Debug helper to test code validation patterns"""
        logger.debug("Testing code validation patterns...")
        
        processed_code = self.preprocess_code(code)
        logger.debug(f"Preprocessed code:\n{processed_code}")
        
        lines = [line.strip() for line in processed_code.split('\n') if line.strip()]
        for line in lines:
            if line.startswith('#'):
                continue
            
            logger.debug(f"\nTesting line: {line}")
            for pattern in self.allowed_patterns:
                match = re.search(pattern, line)
                if match:
                    logger.debug(f"  Matched pattern: {pattern}")

class PhysicsReasoningExecutor:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.execution_history = []
        self.genesis_executor = GenesisSandboxExecutor()
        self.genesis_executor.sandbox = sandbox
        
    def _extract_genesis_code(self, content: str) -> str:
        """Extract Genesis code from content"""
        pattern = r'```genesis\s*(.*?)\s*```'
        matches = re.findall(pattern, content, re.DOTALL)
        return matches[0] if matches else ""
        
    def _parse_reasoning_steps(self, response: str) -> List[Dict[str, str]]:
        """Parse reasoning steps from LLM response"""
        steps = []
        sections = [
            "planning",
            "simulation",
            "observation",
            "reflection",
            "answer"
        ]
        
        for section in sections:
            pattern = f"<{section}>(.*?)</{section}>"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                steps.append({
                    "type": section,
                    "content": match.group(1).strip()
                })
                
        return steps
        
    def execute_reasoning_step(self, step_type: str, content: str) -> Dict:
        """Execute a single reasoning step"""
        if step_type == "simulation":
            # Extract and run Genesis code
            code = self._extract_genesis_code(content)
            return self.genesis_executor.execute_genesis_code(code)
            
        elif step_type == "observation":
            # Get current physics state
            return {
                "object_states": self.sandbox.get_object_states(),
                "measurements": self.sandbox.get_measurements()
            }
            
        return {"status": "success", "step": step_type}
        
    def run_reasoning_cycle(self, prompt: str, context: Dict = None) -> Dict:
        """Run full physics reasoning cycle"""
        try:
            # Stage 1: Understanding
            understanding = self._get_physics_understanding(prompt, context)
            logger.info("\n=== Physics Understanding ===\n%s", understanding.get('choices', [{}])[0].get('message', {}).get('content', ''))
            if "error" in understanding:
                return understanding

            # Stage 2: Planning
            plan = self._create_physics_plan(understanding.get('choices', [{}])[0].get('message', {}).get('content', ''), context)
            logger.info("\n=== Physics Plan ===\n%s", plan.get('choices', [{}])[0].get('message', {}).get('content', ''))
            if "error" in plan:
                return plan

            # Stage 3: Simulation
            simulation = self._run_physics_simulation(plan.get('choices', [{}])[0].get('message', {}).get('content', ''), context)
            logger.info("\n=== Physics Simulation ===\n%s", simulation.get('choices', [{}])[0].get('message', {}).get('content', ''))
            if "error" in simulation:
                return simulation

            # Stage 4: Observation
            observation = self._make_physics_observations(simulation.get('choices', [{}])[0].get('message', {}).get('content', ''), context)
            logger.info("\n=== Physics Observations ===\n%s", observation.get('choices', [{}])[0].get('message', {}).get('content', ''))
            if "error" in observation:
                return observation

            # Stage 5: Reflection
            reflection = self._reflect_on_physics(observation.get('choices', [{}])[0].get('message', {}).get('content', ''), prompt, context)
            logger.info("\n=== Physics Reflection ===\n%s", reflection.get('choices', [{}])[0].get('message', {}).get('content', ''))
            if "error" in reflection:
                return reflection

            # Extract content from responses
            understanding_content = understanding.get('choices', [{}])[0].get('message', {}).get('content', '')
            plan_content = plan.get('choices', [{}])[0].get('message', {}).get('content', '')
            simulation_content = simulation.get('choices', [{}])[0].get('message', {}).get('content', '')
            observation_content = observation.get('choices', [{}])[0].get('message', {}).get('content', '')
            reflection_content = reflection.get('choices', [{}])[0].get('message', {}).get('content', '')

            return {
                "status": "success",
                "understanding": understanding_content,
                "plan": plan_content,
                "simulation": simulation_content,
                "observation": observation_content,
                "reflection": reflection_content,
                "results": [
                    {"type": "understanding", "content": understanding_content},
                    {"type": "simulation", "content": simulation_content},
                    {"type": "observation", "content": observation_content},
                    {"type": "reflection", "content": reflection_content}
                ]
            }

        except Exception as e:
            logger.error(f"Error in physics reasoning cycle: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _get_physics_understanding(self, prompt: str, context: Dict) -> Dict:
        """Get initial physics understanding"""
        understanding_prompt = f"""Analyze this physics concept: "{prompt}"
        
        1. What are the key physical principles involved?
        2. How can we demonstrate this in the physics room?
        3. What measurements or observations would be meaningful?
        
        Explain your understanding and approach."""
        
        return self.sandbox._query_llm_endpoint(understanding_prompt, context)

    def _create_physics_plan(self, understanding: str, context: Dict) -> Dict:
        """Create plan for physics demonstration"""
        if not understanding:
            return {"error": "No understanding provided"}
            
        plan_prompt = f"""Based on this understanding:
        {understanding}
        
        Create a detailed plan for demonstrating this concept:
        1. What objects and configurations do we need?
        2. What interactions should we observe?
        3. What measurements will be important?
        
        Provide a step-by-step plan."""
        
        return self.sandbox._query_llm_endpoint(plan_prompt, context)

    def _run_physics_simulation(self, plan: str, context: Dict) -> Dict:
        """Generate and run physics simulation code"""
        if not plan:
            return {"error": "No plan provided"}
            
        simulation_prompt = f"""Based on this plan:
        {plan}
        
        Generate Genesis physics code to implement this demonstration.
        The code should:
        1. Create a pendulum system
        2. Add necessary measurements
        3. Run the simulation
        
        Use proper Genesis syntax and include all necessary steps."""
        
        return self.sandbox._query_llm_endpoint(simulation_prompt, context)

    def _make_physics_observations(self, simulation: str, context: Dict) -> Dict:
        """Make observations about the physics simulation"""
        if not simulation:
            return {"error": "No simulation provided"}
            
        observation_prompt = f"""Based on the simulation:
        {simulation}
        
        What physical phenomena did we observe?
        What measurements or patterns were significant?
        What insights did this provide?"""
        
        return self.sandbox._query_llm_endpoint(observation_prompt, context)

    def _reflect_on_physics(self, observation: str, original_prompt: str, context: Dict) -> Dict:
        """Reflect on physics understanding gained"""
        if not observation:
            return {"error": "No observations provided"}
            
        reflection_prompt = f"""Given these observations:
        {observation}
        
        And the original question: "{original_prompt}"
        
        1. How did this demonstration help us understand the concept?
        2. What new insights did we gain?
        3. How does this connect to broader physics principles?
        
        Provide a thoughtful reflection."""
        
        return self.sandbox._query_llm_endpoint(reflection_prompt, context)