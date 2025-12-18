import inspect
import jsonref
import asyncio
from typing import (
    Callable,
    List,
    Dict,
    Any,
    Union,
    get_type_hints,
    get_origin,
    get_args,
)
from pydantic import BaseModel, Field, create_model
from defog.llm.models import OpenAIFunctionSpecs, AnthropicFunctionSpecs


def python_type_to_json_schema_type(python_type):
    """Convert Python type to JSON Schema type."""
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    # Handle Optional types
    origin = get_origin(python_type)
    if origin is Union:
        args = get_args(python_type)
        # Check if it's Optional (Union with None)
        if type(None) in args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return python_type_to_json_schema_type(non_none_args[0])

    return type_mapping.get(python_type, "string")


def create_pydantic_model_from_function(func: Callable) -> type[BaseModel]:
    """
    Create a Pydantic model from a regular function's signature.
    """
    sig = inspect.signature(func)
    try:
        # will work for most regular functions
        type_hints = get_type_hints(func)
    except Exception:
        # see if this is a partial function
        try:
            type_hints = get_type_hints(func.func)
        except Exception:
            print("could not get type hints for function. default to Any")
            # if it fails, we'll just use Any for all types
            type_hints = {param_name: Any for param_name in sig.parameters.keys()}

    fields = {}
    for param_name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Get type annotation or default to Any
        param_type = type_hints.get(param_name, Any)

        # Get default value
        if param.default is inspect.Parameter.empty:
            default = ...
        else:
            default = param.default

        # Get description from docstring if available
        docstring = inspect.getdoc(func) or ""
        description = f"Parameter {param_name}"

        # Try to extract parameter descriptions from docstring
        # Simple parsing for common docstring formats
        lines = docstring.split("\n")
        for i, line in enumerate(lines):
            if param_name in line and ":" in line:
                # Found a parameter description
                desc_parts = line.split(":", 1)
                if len(desc_parts) > 1:
                    description = desc_parts[1].strip()
                    break

        fields[param_name] = (
            param_type,
            Field(default=default, description=description),
        )

    # Create a dynamic Pydantic model
    model_name = f"{func.__name__}_Input"
    return create_model(model_name, **fields)


def wrap_regular_function(func: Callable) -> Callable:
    """
    Wrap a regular function to accept a single Pydantic model parameter.
    """
    # Create the Pydantic model for this function
    input_model = create_pydantic_model_from_function(func)

    # Store the original function and model as attributes
    if inspect.iscoroutinefunction(func):

        async def wrapper(input: input_model):
            # Convert the Pydantic model back to kwargs
            kwargs = input.model_dump()
            return await func(**kwargs)
    else:

        def wrapper(input: input_model):
            # Convert the Pydantic model back to kwargs
            kwargs = input.model_dump()
            return func(**kwargs)

    # Copy function metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__dict__["_original_function"] = func
    wrapper.__dict__["_input_model"] = input_model
    wrapper.__annotations__ = {"input": input_model}

    return wrapper


def is_pydantic_style_function(func: Callable) -> bool:
    """
    Check if a function follows the Pydantic style (single parameter named 'input' that is a BaseModel).
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) != 1:
        return False

    param = params[0]
    if param.name != "input":
        return False

    # Check if the annotation is a Pydantic BaseModel
    model_class = param.annotation
    if model_class is inspect.Parameter.empty:
        return False

    return isinstance(model_class, type) and issubclass(model_class, BaseModel)


def cleanup_obj(obj: dict, model: str):
    """
    Converts a pydantic model's json to a format that gemini supports (recursively):
    - Converts properties called "anyOf" to "any_of"
    - Converts all "types" to uppercase
    - Removes "$defs" properties that are created by nested pydantic models
    - Removes $ref properties that are also created by nested pydantic models
    """
    new_obj = obj

    keys = new_obj.keys()
    if "anyOf" in new_obj:
        new_obj["any_of"] = new_obj["anyOf"]
        del new_obj["anyOf"]

    if "type" in new_obj:
        if model.startswith("gemini"):
            new_obj["type"] = new_obj["type"].upper()
        else:
            new_obj["type"] = new_obj["type"]

    if "$defs" in new_obj:
        del new_obj["$defs"]

    if "$ref" in new_obj:
        del new_obj["$ref"]

    for k in keys:
        if isinstance(new_obj[k], dict):
            new_obj[k] = cleanup_obj(new_obj[k], model)

    return new_obj


def get_function_specs(
    functions: List[Callable], model: str
) -> List[Union[OpenAIFunctionSpecs, AnthropicFunctionSpecs]]:
    """Return a list of dictionaries describing each function's name, docstring, and input schema."""
    function_specs = []

    for func in functions:
        # Check if function is already in Pydantic style or needs wrapping
        if not is_pydantic_style_function(func):
            # Wrap regular function to make it compatible
            func = wrap_regular_function(func)

        # Get docstring
        docstring = inspect.getdoc(func) or ""

        # Get the function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        # We assume each function has exactly one parameter that is a Pydantic model
        if len(params) != 1:
            raise ValueError(
                f"Function {func.__name__} does not have exactly one parameter."
            )

        param = params[0]
        model_class = param.annotation  # The Pydantic model

        # Safety check to ensure param.annotation is indeed a Pydantic BaseModel
        if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
            raise ValueError(
                f"Parameter for function {func.__name__} is not a Pydantic model."
            )

        # Get the JSON schema from the model
        input_schema: dict = model_class.model_json_schema()

        # if there are references (due to nested pydantic models), dereference them
        # these show up as "$defs"
        # need proxies=False to replace the refs with actual objects and not just JsonRef instances
        input_schema = jsonref.replace_refs(input_schema, proxies=False)

        # cleanup object
        input_schema = cleanup_obj(input_schema, model)

        # Remove title from input_schema
        input_schema.pop("title")
        # Remove default and title values from input_schema["properties"]
        for prop in input_schema["properties"].values():
            keys_to_remove = [k for k in prop if k in ["default", "title"]]
            for k in keys_to_remove:
                prop.pop(k)
        # Ensure required fields list is present for providers expecting explicit schema
        if "required" not in input_schema:
            input_schema["required"] = list(input_schema["properties"].keys())

        if (
            model.startswith("gpt")
            or model.startswith("o1")
            or model.startswith("chatgpt")
            or model.startswith("o3")
            or model.startswith("o4")
            or model.startswith("deepseek")
        ):
            function_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": docstring,
                        "parameters": input_schema,
                    },
                }
            )
        elif model.startswith("claude") or model.startswith("grok"):
            input_schema["type"] = "object"
            function_specs.append(
                {
                    "name": func.__name__,
                    "description": docstring,
                    "input_schema": input_schema,
                }
            )
        elif model.startswith("gemini"):
            function_specs.append(
                {
                    "type": "function",
                    "name": func.__name__,
                    "description": docstring,
                    "parameters": input_schema,
                }
            )
        elif model.startswith("mistral"):
            function_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": docstring,
                        "parameters": input_schema,
                    },
                }
            )
        else:
            raise ValueError(f"Model does not support function calling: {model}")

    return function_specs


def convert_tool_choice(tool_choice: str, tool_name_list: List[str], model: str):
    """
    Convert a tool choice to a function calling tool choice that is compatible with the model.
    """
    model_map = {
        "openai": {
            "prefixes": ["gpt", "o1", "chatgpt", "o3", "o4", "deepseek"],
            "choices": {
                "auto": "auto",
                "required": "required",
                "any": "required",
                "none": "none",
            },
            "custom": {"type": "function", "function": {"name": tool_choice}},
        },
        "anthropic": {
            "prefixes": ["claude", "grok"],
            "choices": {
                "auto": {"type": "auto"},
                "required": {"type": "any"},
                "any": {"type": "any"},
            },
            "custom": {"type": "tool", "name": tool_choice},
        },
        "gemini": {"prefixes": ["gemini"]},
        "mistral": {
            "prefixes": ["mistral"],
            "choices": {
                "auto": "auto",
                "required": "any",
                "any": "any",
                "none": "none",
            },
            "custom": tool_choice,
        },
    }

    for model_type, config in model_map.items():
        if any(model.startswith(prefix) for prefix in config["prefixes"]):
            if model_type == "gemini":
                from google.genai import types

                config = {
                    "choices": {
                        "auto": types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="AUTO"
                            )
                        ),
                        "required": types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY"
                            )
                        ),
                        "any": types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY"
                            )
                        ),
                        "none": types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="NONE"
                            )
                        ),
                    },
                    "custom": types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY", allowed_function_names=[tool_choice]
                        )
                    ),
                }
            if tool_choice not in config["choices"]:
                # Validate custom tool_choice
                if tool_choice not in tool_name_list:
                    raise ValueError(
                        f"Forced function `{tool_choice}` is not in the list of provided tools"
                    )
                return config["custom"]
            return config["choices"][tool_choice]

    raise ValueError(f"Model `{model}` does not support tools and function calling")


def execute_tool(tool: Callable, inputs: Dict[str, Any]):
    """
    Execute a tool function with the given inputs.
    The inputs are raw values that are expected to match the function's parameter schema.
    Handles both Pydantic-style functions and regular functions.
    """
    if is_pydantic_style_function(tool):
        # Original Pydantic-style function
        sig = inspect.signature(tool)
        model_class = sig.parameters["input"].annotation
        model = model_class.model_validate(inputs)
        return tool(model)
    else:
        # Regular function - call directly with inputs as kwargs
        return tool(**inputs)


async def execute_tool_async(tool: Callable, inputs: Dict[str, Any]):
    """
    Execute a tool function with the given inputs.
    The inputs are raw values that are expected to match the function's parameter schema.
    Handles both Pydantic-style functions and regular functions.
    """
    if is_pydantic_style_function(tool):
        # Original Pydantic-style function
        sig = inspect.signature(tool)
        model_class = sig.parameters["input"].annotation
        model = model_class.model_validate(inputs)
        return await tool(model)
    else:
        # Regular function - call directly with inputs as kwargs
        return await tool(**inputs)


async def execute_tools_parallel(
    tool_calls: List[Dict[str, Any]],
    tool_dict: Dict[str, Callable],
    enable_parallel: bool = False,
) -> List[Any]:
    """
    Execute multiple tool calls either in parallel or sequentially.

    Args:
        tool_calls: List of tool call dictionaries with function name and arguments
        tool_dict: Dictionary mapping function names to callable functions
        enable_parallel: Whether to execute tools in parallel (True) or sequentially (False)

    Returns:
        List of tool execution results in the same order as input tool_calls
    """
    if not enable_parallel:
        # Sequential execution (current behavior)
        results = []
        for tool_call in tool_calls:
            func_name = tool_call.get("function", {}).get("name") or tool_call.get(
                "name"
            )
            func_args = tool_call.get("function", {}).get("arguments") or tool_call.get(
                "arguments", {}
            )

            if func_name in tool_dict:
                tool = tool_dict[func_name]
                if inspect.iscoroutinefunction(tool):
                    result = await execute_tool_async(tool, func_args)
                else:
                    result = execute_tool(tool, func_args)
                results.append(result)
            else:
                results.append(f"Error: Function {func_name} not found")
        return results
    else:
        # Parallel execution
        async def execute_single_tool(tool_call):
            try:
                func_name = tool_call.get("function", {}).get("name") or tool_call.get(
                    "name"
                )
                func_args = tool_call.get("function", {}).get(
                    "arguments"
                ) or tool_call.get("arguments", {})

                if func_name in tool_dict:
                    tool = tool_dict[func_name]
                    if inspect.iscoroutinefunction(tool):
                        return await execute_tool_async(tool, func_args)
                    else:
                        # Run sync function in thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(
                            None, execute_tool, tool, func_args
                        )
                else:
                    return f"Error: Function {func_name} not found"
            except Exception as e:
                return f"Error executing {func_name}: {str(e)}"

        # Execute all tool calls concurrently
        tasks = [execute_single_tool(tool_call) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert any exceptions to error strings
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(f"Error: {str(result)}")
            else:
                processed_results.append(result)

        return processed_results


def verify_post_tool_function(function: Callable):
    """
    Verify that the post_tool_function is a function that takes the required arguments:
    function_name, input_args, tool_result, tool_id
    """
    sig = inspect.signature(function)
    if sig.parameters.get("function_name") is None:
        raise ValueError("post_tool_function must have parameter named `function_name`")
    if sig.parameters.get("input_args") is None:
        raise ValueError("post_tool_function must have parameter named `input_args`")
    if sig.parameters.get("tool_result") is None:
        raise ValueError("post_tool_function must have parameter named `tool_result`")
    if sig.parameters.get("tool_id") is None:
        raise ValueError("post_tool_function must have parameter named `tool_id`")

    return function
