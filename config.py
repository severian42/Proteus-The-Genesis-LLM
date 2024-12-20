import os
import platform
from pathlib import Path
from dotenv import load_dotenv
from typing import Tuple

# Load environment variables
load_dotenv()

class Config:
    # System detection
    IS_APPLE_SILICON = (
        platform.system() == "Darwin" and 
        platform.machine() == "arm64"
    )
    
    @classmethod
    def get_default_backend(cls):
        """Determine the best default backend based on system"""
        if cls.IS_APPLE_SILICON:
            return "metal"  # Default to Metal for Apple Silicon
        try:
            import torch
            if torch.cuda.is_available():
                return "gpu"
        except:
            pass
        return "cpu"  # Fallback to CPU

    # LLM Settings
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gemini-2.0-flash-exp")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    @classmethod
    def validate_config(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in .env file")
        if not cls.OPENAI_BASE_URL:
            raise ValueError("OPENAI_BASE_URL must be set in .env file")

    # Genesis Settings
    _backend = None
    
    @classmethod
    def get_backend(cls):
        if cls._backend is None:
            backend = os.getenv("GENESIS_BACKEND")
            if backend is None:
                backend = cls.get_default_backend()
            cls._backend = cls.validate_backend(backend)
        return cls._backend
    
    GENESIS_PRECISION = os.getenv("GENESIS_PRECISION", "32")
    GENESIS_LOGGING_LEVEL = os.getenv("GENESIS_LOGGING_LEVEL", "debug")
    GENESIS_ASSETS_PATH = Path(os.getenv("GENESIS_ASSETS_PATH", "./assets"))

    # Validation for backend
    @classmethod
    def validate_backend(cls, backend):
        """Validate and possibly adjust backend selection"""
        backend = backend.lower()
        
        if cls.IS_APPLE_SILICON:
            if backend == "gpu":
                print("Warning: GPU backend selected on Apple Silicon. Switching to Metal...")
                return "metal"
            if backend == "metal":
                print("Warning: Metal backend selected on Apple Silicon. Switching to Metal...")
                return "metal"
        elif backend == "metal":
            if not cls.IS_APPLE_SILICON:
                print("Warning: Metal backend selected on non-Apple Silicon. Switching to CPU...")
                return "cpu"
            
        if backend not in ["cpu", "gpu", "metal"]:
            print(f"Warning: Unknown backend '{backend}'. Falling back to CPU...")
            return "cpu"
            
        return backend

    # Visualization Settings
    VIS_CAMERA_RES = eval(os.getenv("VIS_CAMERA_RES", "(1280, 720)"))
    VIS_CAMERA_FOV = int(os.getenv("VIS_CAMERA_FOV", "60"))
    VIS_SHOW_WORLD_FRAME = os.getenv("VIS_SHOW_WORLD_FRAME", "true").lower() == "true"
    VIS_MAX_FPS = int(os.getenv("VIS_MAX_FPS", "60"))

    # Physics Settings
    PHYSICS_GRAVITY = eval(os.getenv("PHYSICS_GRAVITY", "(0, 0, -9.8)"))
    PHYSICS_SUBSTEPS = int(os.getenv("PHYSICS_SUBSTEPS", "10"))
    PHYSICS_ROOM_BOUNDS = eval(os.getenv("PHYSICS_ROOM_BOUNDS", "(-5, -5, -5, 5, 5, 5)"))

    @classmethod
    def get_room_bounds(cls) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Returns room bounds as (lower_bound, upper_bound)"""
        bounds = cls.PHYSICS_ROOM_BOUNDS
        return (bounds[0], bounds[1], bounds[2]), (bounds[3], bounds[4], bounds[5])