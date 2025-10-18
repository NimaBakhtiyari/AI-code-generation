"""
Simple FastAPI Server for Code Generation (Production Ready)

Lightweight API without heavy ML dependencies for quick deployment.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Neuro-Symbolic Code Generation API",
    description="Production API for code generation with template-based approach",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    """Code generation request."""
    prompt: str = Field(..., description="Code generation prompt", min_length=1)
    language: str = Field(default="python", description="Target programming language")
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    """Code generation response."""
    code: str
    language: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    message: str


def generate_python_code(prompt: str) -> str:
    """Generate Python code based on prompt."""
    prompt_lower = prompt.lower()
    
    if "fibonacci" in prompt_lower:
        return '''def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


if __name__ == "__main__":
    for i in range(10):
        print(f"fibonacci({i}) = {fibonacci(i)}")
'''
    
    elif "factorial" in prompt_lower:
        return '''def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


if __name__ == "__main__":
    for i in range(10):
        print(f"{i}! = {factorial(i)}")
'''
    
    elif "prime" in prompt_lower:
        return '''def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def get_primes(limit: int) -> list[int]:
    """Get all prime numbers up to limit."""
    return [n for n in range(2, limit + 1) if is_prime(n)]


if __name__ == "__main__":
    primes = get_primes(100)
    print(f"Primes up to 100: {primes}")
'''
    
    elif "sort" in prompt_lower or "bubble" in prompt_lower:
        return '''def bubble_sort(arr: list) -> list:
    """Sort array using bubble sort algorithm."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


if __name__ == "__main__":
    data = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {data}")
    sorted_data = bubble_sort(data.copy())
    print(f"Sorted: {sorted_data}")
'''
    
    elif "binary search" in prompt_lower:
        return '''def binary_search(arr: list, target: int) -> int:
    """Binary search algorithm - returns index or -1."""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    result = binary_search(data, target)
    print(f"Found {target} at index: {result}")
'''
    
    elif "class" in prompt_lower:
        return f'''class GeneratedClass:
    """
    {prompt}
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        return f"Hello from {{self.name}}"


if __name__ == "__main__":
    obj = GeneratedClass("Example")
    print(obj.greet())
'''
    
    else:
        return f'''def generated_function():
    """
    {prompt}
    """
    # TODO: Implement function logic based on prompt
    pass


if __name__ == "__main__":
    generated_function()
'''


def generate_javascript_code(prompt: str) -> str:
    """Generate JavaScript code based on prompt."""
    return f'''// {prompt}

function generatedFunction(param) {{
    // TODO: Implement function logic
    console.log("Function called with:", param);
    return param;
}}

// Example usage
const result = generatedFunction("test");
console.log("Result:", result);

module.exports = {{ generatedFunction }};
'''


def generate_java_code(prompt: str) -> str:
    """Generate Java code based on prompt."""
    return f'''// {prompt}

public class GeneratedClass {{
    public static void main(String[] args) {{
        System.out.println("Generated Java code");
        Object result = generatedMethod("test");
        System.out.println("Result: " + result);
    }}
    
    public static Object generatedMethod(Object param) {{
        // TODO: Implement method logic
        return param;
    }}
}}
'''


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Neuro-Symbolic Code Generation API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        message="API is running with template-based code generation",
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_code(request: GenerateRequest):
    """
    Generate code based on prompt.
    
    Uses template-based generation for common patterns.
    """
    try:
        logger.info(
            "code_generation_requested",
            prompt_length=len(request.prompt),
            language=request.language,
        )
        
        if request.language.lower() == "python":
            code = generate_python_code(request.prompt)
        elif request.language.lower() in ["javascript", "js"]:
            code = generate_javascript_code(request.prompt)
        elif request.language.lower() == "java":
            code = generate_java_code(request.prompt)
        else:
            code = generate_python_code(request.prompt)
        
        metadata = {
            "generation_method": "template_based",
            "language": request.language,
            "prompt_tokens": len(request.prompt.split()),
        }
        
        logger.info(
            "code_generation_completed",
            code_length=len(code),
        )
        
        return GenerateResponse(
            code=code,
            language=request.language,
            metadata=metadata,
        )
        
    except Exception as e:
        logger.error("code_generation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code generation failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "neurosymbolic_codegen.api.simple_main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info",
    )
