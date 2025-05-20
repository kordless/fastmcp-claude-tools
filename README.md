# FastMCP with Claude Tool Integration Example

This repository demonstrates how to implement a web server that integrates Claude's tools capabilities with the FastMCP (Model Control Protocol) framework. It provides a complete example of defining custom tools that Claude can use to perform calculations and retrieve information.

## Key Features

- **Claude Tool Integration**: Uses Claude's tool use capabilities with proper conversation flow and tool_use_id matching
- **FastMCP Tool Definitions**: Utilizes FastMCP to define and manage tool schemas and implementations
- **Asynchronous Web Server**: Built with Quart for high-performance async handling of requests
- **Complete Conversation Flow**: Demonstrates the full conversation loop with tool calls and tool results
- **Error Handling**: Robust error handling for tool execution and API communication
- **Example Tools**: Includes sample calculator and weather lookup tools

## How It Works

1. **Tool Definition**: Tools are defined using the FastMCP decorators, which automatically generate the appropriate schemas
2. **API Integration**: The server converts FastMCP tool definitions to Claude's expected format
3. **Conversation Flow**: Manages the complete conversation flow, including preserving tool_use_id references
4. **Tool Execution**: Executes the appropriate tool functions when Claude calls them
5. **Result Handling**: Formats and returns tool results in the format Claude expects

## Getting Started

### Prerequisites

- Python 3.8+
- Anthropic API key
- Docker (optional, for running with PostgreSQL for the memory example)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fastmcp-claude-tools.git
cd fastmcp-claude-tools
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Anthropic API key as an environment variable:

**Windows PowerShell:**
```powershell
$env:ANTHROPIC_API_KEY="your-api-key-here"
```

**Windows CMD:**
```cmd
set ANTHROPIC_API_KEY=your-api-key-here
```

**Linux/macOS:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Running the Example

1. Start the server:
```bash
python app.py
```

2. In another terminal, run the client:
```bash
python client.py
```

## Implementation Details

The repository demonstrates several key aspects of integrating Claude with FastMCP:

1. **Tool Schema Generation**: The application properly extracts tool schemas from FastMCP, avoiding hardcoded definitions
2. **Proper Tool Use Handling**: Implements the Claude API's expected format for tool use and tool results
3. **Conversation Context Preservation**: Maintains the conversation context with proper tool_use references
4. **Error Handling**: Gracefully handles errors in tool execution and API communication
5. **Python Type Hints**: Uses Python's type hints for automatic schema generation

## Important Components

- **app.py**: Main server implementation with tool definitions and Claude API integration
- **client.py**: Simple client to demonstrate how to interact with the server
- **memory_app.py** and **memory_system.py**: Advanced example with a memory system (optional)

## Claude Tool Use Control

The implementation demonstrates various methods to control Claude's tool use behavior:

1. **Tool Choice**: Configure whether Claude should use tools, which tools to use, etc.
2. **Chain of Thought**: Encourage Claude to explain its reasoning in thinking tags
3. **Tool Use Structure**: Properly structure the conversation to maintain tool context

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Anthropic](https://www.anthropic.com/) for Claude API
- [FastMCP](https://fastmcp.readthedocs.io/) for the Model Control Protocol framework
- [Quart](https://pgjones.gitlab.io/quart/) for the async web framework
