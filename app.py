"""
Sample FastMCP with Quart application that integrates with Claude
"""

import asyncio
import json
import os
import logging
import pprint
from typing import Dict, Any, List

from quart import Quart, request, jsonify
from anthropic import AsyncAnthropic
from fastmcp import FastMCP

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fastmcp_app")

# Initialize Quart app
app = Quart(__name__)
logger.info("Initializing Quart app")

# Initialize FastMCP with the name of your tool namespace
mcp = FastMCP("Sample Tools")
logger.info("FastMCP initialized")

# Define some sample tools using FastMCP
@mcp.tool()
async def calculator(expression: str) -> Dict[str, Any]:
    """
    Calculates mathematical expressions.
    
    Args:
        expression: Mathematical expression (e.g., "2 + 3 * 4")
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def weather_lookup(location: str) -> Dict[str, Any]:
    """
    Looks up weather information for a given location.
    
    Args:
        location: Name of the city or location
    """
    # This is a mock implementation
    weather_data = {
        "New York": {"temperature": 72, "condition": "Sunny"},
        "London": {"temperature": 65, "condition": "Cloudy"},
        "Tokyo": {"temperature": 80, "condition": "Rainy"},
    }
    
    if location in weather_data:
        return {"success": True, "data": weather_data[location]}
    else:
        return {"success": False, "error": f"No weather data available for {location}"}

# Function to process Claude's tool calls using mcp tool functions
async def process_tool_calls(tool_calls):
    logger.info(f"Processing {len(tool_calls)} tool calls")
    results = []
    
    for idx, tool_call in enumerate(tool_calls):
        logger.debug(f"Tool call {idx}: {tool_call}")
        logger.debug(f"Tool call type: {type(tool_call)}")
        logger.debug(f"Tool call dir: {dir(tool_call)}")
        
        try:
            # Get tool_use_id for proper response formatting - this is CRITICAL
            tool_use_id = None
            if hasattr(tool_call, 'id'):
                tool_use_id = tool_call.id
                logger.info(f"Extracted tool_use_id from attribute: {tool_use_id}")
            elif isinstance(tool_call, dict) and 'id' in tool_call:
                tool_use_id = tool_call['id']
                logger.info(f"Extracted tool_use_id from dict: {tool_use_id}")
            
            # If we still don't have an ID, log an error as this will cause problems
            if not tool_use_id:
                logger.error(f"Could not extract tool_use_id from tool call: {tool_call}")
                logger.error("Tool result will likely fail as each tool_result must have a matching tool_use_id")
            
            # Try different approaches to extract tool call data based on actual structure
            if hasattr(tool_call, 'name') and hasattr(tool_call, 'input'):
                # Handle ToolUseBlock objects from newer Claude API
                tool_name = tool_call.name
                tool_params = tool_call.input  # This is already a dict
                logger.info(f"Extracted tool name from ToolUseBlock: {tool_name}")
                logger.debug(f"Tool parameters (direct): {tool_params}")
                # Skip parameter parsing as it's already a dict
                tool_params_str = None
            elif hasattr(tool_call, 'name') and hasattr(tool_call, 'parameters'):
                # Handle older API format
                tool_name = tool_call.name
                tool_params_str = tool_call.parameters
                logger.info(f"Extracted tool name: {tool_name}")
                logger.debug(f"Tool parameters (raw): {tool_params_str}")
            else:
                # Fall back to dictionary access if attributes aren't found
                tool_name = tool_call["name"] if "name" in tool_call else "unknown"
                tool_params_str = tool_call["parameters"] if "parameters" in tool_call else "{}"
                logger.info(f"Extracted tool name (dict method): {tool_name}")
            
            # Parse the parameters if not done yet
            if tool_params_str is not None:
                try:
                    tool_params = json.loads(tool_params_str)
                    logger.debug(f"Parsed parameters: {tool_params}")
                except (TypeError, json.JSONDecodeError) as e:
                    logger.error(f"Error parsing parameters: {e}")
                    if isinstance(tool_params_str, dict):
                        tool_params = tool_params_str  # Already a dict
                    else:
                        tool_params = {}  # Default to empty dict if we can't parse
        
            # Try to find the tool by getting the function directly
            if tool_name == "calculator":
                logger.info(f"Executing calculator with params: {tool_params}")
                result = await calculator(**tool_params)
            elif tool_name == "weather_lookup":
                logger.info(f"Executing weather_lookup with params: {tool_params}")
                result = await weather_lookup(**tool_params)
            else:
                raise ValueError(f"Tool {tool_name} not found")
                
            # Format the result for Claude
            logger.debug(f"Tool execution result: {result}")
            formatted_result = {
                "name": tool_name,
                "output": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                "tool_use_id": tool_use_id if tool_use_id else f"toolu_{tool_name}_{idx}"
            }
            logger.info(f"Formatted result: {formatted_result}")
            results.append(formatted_result)
            
        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            error_result = {
                "name": tool_name if 'tool_name' in locals() else "unknown_tool",
                "output": json.dumps({"error": f"Error executing tool: {str(e)}"}),
                "tool_use_id": tool_use_id if 'tool_use_id' in locals() else f"toolu_error_{idx}"
            }
            logger.info(f"Error result: {error_result}")
            results.append(error_result)
    
    logger.info(f"Returning {len(results)} tool results")
    return results

# Convert FastMCP tools to Claude's format
async def get_claude_tools():
    # Get the tool schemas directly from FastMCP - it's an async function
    try:
        logger.info("Calling mcp.get_tools()")
        mcp_tools = await mcp.get_tools()
        logger.info(f"Raw mcp_tools: {mcp_tools}")
        claude_tools = []
        
        if mcp_tools is None:
            logger.warning("mcp.get_tools() returned None")
            return []
            
        if not isinstance(mcp_tools, list):
            logger.warning(f"mcp.get_tools() returned non-list type: {type(mcp_tools)}")
            # Try to convert to list if possible
            try:
                mcp_tools = list(mcp_tools)
                logger.info(f"Converted mcp_tools to list, now length: {len(mcp_tools)}")
            except:
                logger.error("Could not convert mcp_tools to list")
                return []
        
        logger.info(f"Converting {len(mcp_tools)} FastMCP tools to Claude format")
    except Exception as e:
        logger.error(f"Error getting tools from FastMCP: {e}")
        return []
    
    # Debug the actual structure of the tools
    logger.debug(f"MCP tools structure: {type(mcp_tools)}")
    
    # Handle the case where mcp_tools is a dictionary of tools (which appears to be the actual structure)
    if isinstance(mcp_tools, dict):
        logger.info("MCP tools is a dictionary, extracting tools properly")
        claude_tools = []
        
        for tool_name, tool_obj in mcp_tools.items():
            # Extract the tool information from the Tool object
            try:
                tool_desc = getattr(tool_obj, "description", "")
                tool_params = getattr(tool_obj, "parameters", {})
                
                claude_tool = {
                    "name": tool_name,
                    "description": tool_desc,
                    "input_schema": tool_params
                }
                
                claude_tools.append(claude_tool)
                logger.info(f"Added tool {tool_name} from dictionary")
            except Exception as e:
                logger.error(f"Error extracting tool {tool_name}: {e}")
        
        return claude_tools
    
    # For list type tools
    if isinstance(mcp_tools, list):
        try:
            for tool in mcp_tools:
                if isinstance(tool, dict):
                    # Check if it's already in the right format
                    if "name" in tool and "description" in tool:
                        claude_tool = {
                            "name": tool["name"],
                            "description": tool["description"],
                            "input_schema": tool.get("parameters", {})
                        }
                        claude_tools.append(claude_tool)
                        logger.debug(f"Converted tool: {claude_tool['name']}")
                    else:
                        # Log the structure for debugging
                        logger.debug(f"Unknown tool structure: {tool}")
                elif isinstance(tool, str):
                    # Just a name, log and skip
                    logger.warning(f"Tool '{tool}' is just a string name without schema information")
                else:
                    # Other unexpected type
                    logger.warning(f"Unexpected tool type: {type(tool)}")
        except Exception as e:
            logger.error(f"Error converting list tools: {e}")
    
    # If we couldn't convert any tools, log an error but don't fall back to hardcoded tools
    if not claude_tools:
        logger.error("No tools converted successfully. Check FastMCP integration!")
        logger.error("The app may not work correctly, but we won't hardcode tools")
    
    return claude_tools

@app.route("/chat", methods=["POST"])
async def chat():
    data = await request.get_json()
    
    messages = data.get("messages", [])
    max_iterations = data.get("max_iterations", 3)  # Default to 3 iterations
    
    logger.info(f"Starting chat with {len(messages)} messages, max_iterations={max_iterations}")
    
    # Initialize the Claude client
    api_key = os.environ.get("ANTHROPIC_API_KEY", "your_api_key_here")
    client = AsyncAnthropic(api_key=api_key)
    
    # Get Claude-compatible tool definitions
    claude_tools = await get_claude_tools()
    logger.debug(f"Using Claude tools: {json.dumps(claude_tools, indent=2)}")
    
    try:
        iteration = 0
        final_response = None
        has_used_tools = False
        final_message_sent = False
        
        while iteration < max_iterations:
            logger.info(f"Starting iteration {iteration+1}/{max_iterations}")
            
            # If this is the last iteration and we've used tools, add a final message
            if iteration == max_iterations - 1 and has_used_tools and not final_message_sent:
                messages.append({
                    "role": "user",
                    "content": "Is there anything else to add? Only respond if you have additional information to share."
                })
                final_message_sent = True
                logger.info("Added final 'anything else?' message")
            
            # Add detailed logging for tools
            logger.info(f"Got claude_tools: {claude_tools}")
            logger.info(f"Type of claude_tools: {type(claude_tools)}")
            logger.info(f"Length of claude_tools: {len(claude_tools) if isinstance(claude_tools, list) else 'not a list'}")
            
            # Prepare API parameters
            api_params = {
                "model": "claude-3-5-sonnet-20240620",
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0,
                "system": "You have access to helpful tools. Use them when necessary. Before using tools, explain your reasoning step-by-step in <thinking> tags."
            }
            
            # Only add tools and tool_choice if we have valid tools
            if claude_tools and isinstance(claude_tools, list) and len(claude_tools) > 0:
                logger.info("Adding tools and tool_choice to API params")
                api_params["tools"] = claude_tools
                api_params["tool_choice"] = {"type": "auto"}
            else:
                logger.warning("No valid tools found, omitting tools and tool_choice parameters")
            
            # Log the complete API request for debugging
            logger.debug(f"API request parameters: {json.dumps(api_params, default=str)}")
            
            # Call Claude with tools
            logger.debug(f"Calling Claude with {len(messages)} messages")
            response = await client.messages.create(**api_params)
            
            # Debug: Log the response structure
            logger.debug(f"Claude response type: {type(response)}")
            logger.debug(f"Claude response attributes: {dir(response)}")
            logger.debug(f"Response content type: {type(response.content)}")
            logger.debug(f"First content item type: {type(response.content[0])}")
            logger.debug(f"First content item attributes: {dir(response.content[0])}")
            
            # Store the current response content
            current_content = response.content[0].text
            logger.debug(f"Claude response text: {current_content[:100]}...")
            
            # Check for tool calls in the content
            tool_calls = []
            tool_use_blocks = []  # Store the actual tool_use blocks for reference
            
            for idx, content_item in enumerate(response.content):
                logger.debug(f"Checking content item {idx}, type: {type(content_item)}")
                
                if hasattr(content_item, 'type'):
                    logger.debug(f"Content item type: {content_item.type}")
                
                if hasattr(content_item, 'tool_calls') and content_item.tool_calls:
                    logger.info(f"Found tool_calls in content item {idx}: {content_item.tool_calls}")
                    tool_calls.extend(content_item.tool_calls)
                elif hasattr(content_item, 'type') and content_item.type == 'tool_use':
                    # For tool_use blocks, we need both the tool call and the block itself
                    logger.info(f"Found tool_use content: {content_item}")
                    # Add the tool_use block to tool_calls for processing
                    tool_calls.append(content_item)
                    # Also store the original block for preserving in conversation
                    tool_use_blocks.append(content_item)
            
            # If there are no tool calls, we're done
            if not tool_calls:
                logger.info("No tool calls found, ending conversation")
                final_response = response
                break
                
            # Log the tool calls
            logger.info(f"Found {len(tool_calls)} tool calls")
            for tc_idx, tc in enumerate(tool_calls):
                logger.debug(f"Tool call {tc_idx} type: {type(tc)}")
                logger.debug(f"Tool call {tc_idx} attributes: {dir(tc)}")
                if hasattr(tc, 'name'):
                    logger.info(f"Tool call {tc_idx}: {tc.name}")
                else:
                    logger.info(f"Tool call {tc_idx} structure: {tc}")
                
            # Process the tool calls
            logger.info("Processing tool calls")
            tool_results = await process_tool_calls(tool_calls)
            has_used_tools = True
            logger.debug(f"Tool results: {tool_results}")
            
            # If we're at the last iteration but have tool calls, extend by 1
            if iteration == max_iterations - 1:
                logger.info("Extending iteration limit by 1 to process tool results")
                max_iterations += 1
            
            # Add the assistant's response with tool calls to the conversation
            # Convert tool_calls objects to dictionaries for JSON serialization
            tool_calls_dicts = []
            for tc in tool_calls:
                try:
                    # Handle ToolUseBlock objects
                    if hasattr(tc, 'input') and hasattr(tc, 'name'):
                        tool_call_dict = {
                            "name": tc.name,
                            "parameters": tc.input
                        }
                    # Handle older format
                    elif hasattr(tc, 'name') and hasattr(tc, 'parameters'):
                        tool_call_dict = {
                            "name": tc.name,
                            "parameters": tc.parameters
                        }
                    else:
                        # Unknown format, try to extract fields
                        tool_call_dict = {}
                        if hasattr(tc, 'name'):
                            tool_call_dict["name"] = tc.name
                        if hasattr(tc, 'input'):
                            tool_call_dict["parameters"] = tc.input
                        elif hasattr(tc, 'parameters'):
                            tool_call_dict["parameters"] = tc.parameters
                    
                    tool_calls_dicts.append(tool_call_dict)
                    logger.debug(f"Converted tool call: {tool_call_dict}")
                except Exception as e:
                    logger.error(f"Error converting tool call to dict: {e}")
                    logger.debug(f"Problem tool call object: {tc}")
                    # Try a different approach based on the actual structure
                    tool_call_dict = {"error": "Could not convert tool call"}
                    if hasattr(tc, "__dict__"):
                        tool_call_dict = tc.__dict__
                    tool_calls_dicts.append(tool_call_dict)
                
            # According to Anthropic docs, when sending conversation history:
            # 1. Assistant messages must preserve the original content structure including tool_use blocks
            # 2. Tool results should be added as tool_result content blocks in user messages
            
            # We need to preserve the original response structure with tool_use blocks
            assistant_content = []
            
            # First, add any text content
            for content_item in response.content:
                if hasattr(content_item, 'type') and content_item.type == 'text':
                    assistant_content.append({
                        "type": "text",
                        "text": content_item.text
                    })
                elif hasattr(content_item, 'type') and content_item.type == 'tool_use':
                    # Preserve the tool_use blocks
                    assistant_content.append({
                        "type": "tool_use",
                        "id": content_item.id,
                        "name": content_item.name,
                        "input": content_item.input
                    })
            
            # If we couldn't extract structured content, fall back to text-only
            if not assistant_content:
                assistant_content = current_content
                
            assistant_message = {
                "role": "assistant",
                "content": assistant_content
            }
            messages.append(assistant_message)
            
            # Add the tool results to the conversation as tool_result content blocks
            if tool_results:
                # Format tool results as content blocks
                tool_result_blocks = []
                for tool_result in tool_results:
                    # This is critical: we must use the exact tool_use_id from the Claude response
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tool_result["tool_use_id"],
                        "content": tool_result["output"]
                    })
                
                user_message = {
                    "role": "user",
                    "content": tool_result_blocks
                }
                messages.append(user_message)
            
            logger.info(f"Completed iteration {iteration+1}")
            iteration += 1
        
        # If we've reached max iterations without a final response, use the last one
        if not final_response:
            logger.info("Reached max iterations, using last response")
            final_response = response
        
        # Format the final response
        result = {
            "response": final_response.content[0].text,
            "conversation": messages,
            "tool_calls_used": has_used_tools,
            "iterations": iteration
        }
        
        logger.info(f"Returning final response after {iteration} iterations")
        logger.debug(f"Final response length: {len(result['response'])}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/tools", methods=["GET"])
async def list_tools():
    """Endpoint to list available tools"""
    logger.info("Tools endpoint called")
    
    tools = await get_claude_tools()
    logger.debug(f"Returning {len(tools)} tools")
    
    return jsonify(tools)

if __name__ == "__main__":
    logger.info(f"Starting server on http://127.0.0.1:5111")
    app.run(host="127.0.0.1", port=5111)
