"""
Specific tool implementations
"""

from agent.tool.tools.search_tool import SearchTool
from agent.tool.tools.calculator_tool import CalculatorTool
from agent.tool.tools.wiki_search_tool import WikiSearchTool
__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
] 

def _default_tools(env, tool_config_path=None):
    if env == 'search':
        return [SearchTool(config_name=tool_config_path)]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    else:
        raise NotImplementedError
