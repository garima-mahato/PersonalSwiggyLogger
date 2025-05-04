from typing import Dict, Any, Union
from pydantic import BaseModel
from mcp import ClientSession
import ast
import logging

# Optional: import log from agent if shared, else define locally
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

# # Configure logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # File handler
# file_handler = logging.FileHandler("logs/action.log")
# file_handler.setLevel(logging.INFO)
# file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(file_formatter)

# # Console handler
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_formatter = logging.Formatter('%(levelname)s: %(message)s')
# console_handler.setFormatter(console_formatter)

# # Add handlers to logger
# logger.addHandler(file_handler)
# logger.addHandler(console_handler)


class ToolCallResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Union[str, list, dict]
    raw_response: Any


def parse_function_call(response: str) -> tuple[str, Dict[str, Any]]:
    """Parses FUNCTION_CALL string into tool name and arguments."""
    try:
        if not response.startswith("FUNCTION_CALL:"):
            raise ValueError("Not a valid FUNCTION_CALL")

        _, function_info = response.split(":", 1)
        parts = [p.strip() for p in function_info.split("|")]
        func_name, param_parts = parts[0], parts[1:]

        result = {}
        for part in param_parts:
            if "=" not in part:
                raise ValueError(f"Invalid param: {part}")
            key, value = part.split("=", 1)

            try:
                parsed_value = ast.literal_eval(value)
            except Exception:
                parsed_value = value.strip()

            # Handle nested keys
            keys = key.split(".")
            current = result
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = parsed_value

        log("parser", f"Parsed: {func_name} → {result}")
        return func_name, result

    except Exception as e:
        log("parser", f"❌ Failed to parse FUNCTION_CALL: {e}")
        raise


async def execute_tool(session: ClientSession, tools: list[Any], response: str) -> ToolCallResult:
    """Executes a FUNCTION_CALL via MCP tool session."""
    try:
        tool_name, arguments = parse_function_call(response)

        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registered tools")

        log("tool", f"⚙️ Calling '{tool_name}' with: {arguments}")
        result = await session.call_tool(tool_name, arguments=arguments)

        if hasattr(result, 'content'):
            if isinstance(result.content, list):
                out = [getattr(item, 'text', str(item)) for item in result.content]
            else:
                out = getattr(result.content, 'text', str(result.content))
        else:
            out = str(result)

        log("tool", f"✅ {tool_name} result: {out}")
        return ToolCallResult(
            tool_name=tool_name,
            arguments=arguments,
            result=out,
            raw_response=result
        )

    except Exception as e:
        log("tool", f"⚠️ Execution failed for '{response}': {e}")
        raise
