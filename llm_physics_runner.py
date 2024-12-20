from llm_physics_sandbox import LLMPhysicsSandbox
import logging
import json
import time
from typing import List, Dict, Any
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.traceback import install
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback

# Install rich traceback handler
install(show_locals=True)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
console = Console()

# Initialize FastAPI
app = FastAPI(title="Genesis Physics Sandbox")

class PhysicsQuery(BaseModel):
    prompt: str
    context: dict = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

@app.post("/query")
async def handle_query(query: PhysicsQuery):
    try:
        sandbox = LLMPhysicsSandbox(visualize=True)
        response = sandbox.query_llm(query.prompt, query.context)
        
        if "error" in response:
            error_message = f"""
Failed to execute physics demonstration.
Error: {response['error']}

Original plan was:
{response.get('plan', 'No plan available')}

Last error encountered:
{response.get('last_error', 'Unknown error')}
"""
            console.print(f"\n[bold red]Error:[/bold red] {error_message}")
            return {"status": "error", "message": error_message}
            
        # Handle successful execution
        success_message = f"""
Physics demonstration completed successfully in {response.get('attempts', 1)} attempts.

Plan:
{response['plan']}

Executed code:
```python
{response['code']}
```

Results Summary:
{json.dumps(response['execution_results'], indent=2)}

Conclusion:
{response['conclusion']}
"""
        console.print(Markdown(success_message))
        return {"status": "success", "message": success_message}
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def display_welcome_message():
    welcome_text = """
# Genesis LLM Physics Sandbox

Welcome to the Genesis Physics Sandbox! You can chat naturally with the AI that has access to a 4D physics environment.
The AI can:
- Create and manipulate physical objects
- Run physics simulations to test ideas
- Use the environment to reason about problems
- Transform the space to explore concepts

Just chat naturally - the AI will use the physics sandbox when helpful for reasoning or demonstration.

Type 'exit' to quit, 'help' for commands, or just start chatting!
"""
    console.print(Markdown(welcome_text))

def main():
    if __name__ == "__main__":
        logger.info("Starting LLM Physics Sandbox...")
        sandbox = LLMPhysicsSandbox(visualize=True)
        display_welcome_message()
        
        # Initialize conversation history
        conversation_history = [
            {"role": "system", "content": """You are an AI with access to a 4D physics sandbox environment.
            You can use this environment to think through problems, demonstrate concepts, and explore ideas.
            When appropriate, you can create and manipulate objects in the sandbox to help explain or reason about things.
            Respond naturally to the user while leveraging your ability to experiment in the physics environment."""}
        ]
        
        try:
            while True:
                # Get user input
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                    
                # Add user message to history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Get LLM's response
                response = sandbox.query_llm(user_input, context={
                    "conversation_history": conversation_history,
                    "sandbox_state": sandbox._get_state() if hasattr(sandbox.scene, 'get_state') else {}
                })
                
                # Check for errors in response
                if "error" in response:
                    logger.error(f"Error in LLM response: {response['error']}")
                    console.print(f"\n[bold red]Error:[/bold red] {response['error']}")
                    continue
                
                # Parse any physics actions from the response
                actions = sandbox.parse_llm_response(response)
                logger.debug(f"Parsed actions: {actions}")
                
                # Execute physics actions if any
                for action in actions:
                    logger.debug(f"Executing action: {action}")
                    result = sandbox.execute_llm_action(action)
                    logging.debug(f"Action result: {result}")
                    
                    # Add execution result to conversation history
                    if result.get("status") == "success":
                        conversation_history.append({
                            "role": "system",
                            "content": f"Physics simulation result: {json.dumps(result['state'])}"
                        })
                    else:
                        conversation_history.append({
                            "role": "system",
                            "content": f"Error executing physics code: {result.get('error', 'Unknown error')}"
                        })
                    
                    # Run simulation steps to show effects
                    for _ in range(10):
                        try:
                            state = sandbox._get_state()
                        except Exception as e:
                            logging.debug(f"Could not get state during simulation: {str(e)}")
                            state = {}
                        time.sleep(0.01)  # Small delay to visualize
                
                # Extract and display LLM's text response
                llm_message = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                console.print("\n[bold green]AI:[/bold green]", style="bold green")
                console.print(Markdown(llm_message))
                
                # Add AI response to history
                conversation_history.append({"role": "assistant", "content": llm_message})
                
        except KeyboardInterrupt:
            console.print("\nGoodbye! Thanks for chatting!")
        except Exception as e:
            logging.error(f"Error in chat loop: {str(e)}")
            raise
        finally:
            logging.info("Chat session ended")

if __name__ == "__main__":
    main() 