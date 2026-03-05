# """
# Search tool implementation for simulating internet searches
# """

# import time
# import random
# import os
# from typing import Dict, List, Any, Optional

# from agent.tool.tool_base import Tool

# # from txtai.embeddings import Embeddings
# import faiss
# from FlagEmbedding import FlagAutoModel
# import json
# import requests

# class SearchTool(Tool):
#     """
#     Tool for simulating internet searches using the NeuML/txtai-wikipedia model
#     """
    
#     def __init__(self, config_name: Optional[str] = None):
#         """
#         Initialize the search tool
        
#         Args:
#             search_db: Custom search database, if None, use default
#         """
#         name = "search"
#         description = "Search for information on the internet using Wikipedia as a knowledge source."
#         parameters = {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "Search query"
#                 },
#                 # "limit": {
#                 #     "type": "integer",
#                 #     "description": "Maximum number of results to return (default: 5)"
#                 # }
#             },
#             "required": ["query"]
#         }
        
#         super().__init__(name, description, parameters)
#         self.config_name = config_name
#         self.config = self._load_config()
#         self.dataset_ports = self.config.get("dataset_ports", {})
        
#         # 构建 API URLs
#         self.search_api_urls = {}
#         for dataset, port in self.dataset_ports.items():
#             self.search_api_urls[dataset] = f"http://localhost:{port}/search"
        
#         self._print_init_info()

#     def _print_init_info(self):
#         """打印初始化信息"""
#         print(f"\n{'='*60}")
#         print(f"SearchTool 初始化")
#         print(f"{'='*60}")
#         print(f"数据集端口映射:")
#         for dataset, port in self.dataset_ports.items():
#             print(f"  {dataset:20s}: port {port} -> {self.search_api_urls[dataset]}")
#         print(f"{'='*60}\n")

#     def execute(self, args: Dict) -> str:
#         """
#         Execute search query
        
#         Args:
#             args: Tool parameters, containing:
#                 - "query": search query string
#                 - "limit": optional int to limit number of results
            
#         Returns:
#             Formatted search results
#         """
#         pass
    
#     def batch_execute(self, args_list: List[Dict]) -> List[str]:
#         """
#         批量执行搜索查询，根据 data_source 自动路由到不同端口
        
#         Args:
#             args_list: 参数列表，每个元素包含:
#                 - "query": 查询字符串
#                 - "data_source": 数据集名称（必需）
            
#         Returns:
#             搜索结果列表
#         """
#         queries = [x["query"] for x in args_list]
#         data_sources = [x["data_source"] for x in args_list]
        
#         # 按数据源分组，调用不同的 API
#         return self._batch_call_by_datasource(queries, data_sources)

#     def _batch_call_by_datasource(self, queries: List[str], data_sources: List[str]) -> List[str]:
#         """
#         根据数据源分组，调用不同的 API 端口
        
#         Args:
#             queries: 查询列表
#             data_sources: 对应的数据源列表
            
#         Returns:
#             结果列表（保持原始顺序）
#         """
#         # 按数据源分组
#         source_groups = {}  # {data_source: [(index, query), ...]}
        
#         for idx, (query, ds) in enumerate(zip(queries, data_sources)):
#             if ds not in source_groups:
#                 source_groups[ds] = []
#             source_groups[ds].append((idx, query))
        
#         # 初始化结果列表
#         results = [None] * len(queries)
        
#         # 对每个数据源分别调用对应的 API
#         for ds_key, group in source_groups.items():
#             indices = [item[0] for item in group]
#             ds_queries = [item[1] for item in group]
            
#             # 获取对应的 API URL
#             api_url = self.get_api_url(ds_key)
            
#             print(f"  批量调用 [{ds_key}]: {len(ds_queries)} 个查询 -> {api_url}")
            
#             try:
#                 response = requests.post(api_url, json={"queries": ds_queries}, timeout=120)
#                 response.raise_for_status()
#                 ds_results = response.json()
                
#                 # 将结果放回原始位置
#                 for result, orig_idx in zip(ds_results, indices):
#                     results[orig_idx] = result
                    
#             except Exception as e:
#                 print(f"⚠ 调用 [{ds_key}] API 失败: {e}")
#                 # 对失败的查询填充错误信息
#                 for orig_idx in indices:
#                     results[orig_idx] = f"Error: {str(e)}"
        
#         return results
    
#     def calculate_reward(self, args: Dict, result: str) -> float:
#         """
#         Calculate reward for search action
        
#         Args:
#             args: Tool parameters
#             result: Tool execution result
            
#         Returns:
#             Reward value
#         """
#         # valid tool call
#         if "results" in result:
#             return 0.0
#         else:
#             return 0.0
            
#     def get_api_port(self):
#         """读取配置文件中的API端口设置"""
#         # 获取项目根目录，假设 agent 模块在项目根目录下
#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
#         config_path = os.path.join(project_root, "config.json")
        
#         try:
#             with open(config_path, "r") as f:
#                 config = json.load(f)
#                 return config.get("search_api_port", 8002)  # 默认为8002
#         except (FileNotFoundError, json.JSONDecodeError) as e:
#             print(f"警告: SearchTool无法读取配置文件: {e}，使用默认端口8002")
#             return 8002  # 默认为8002

#     def get_api_url(self, data_source: str) -> str:
#         """
#         根据数据源获取对应的 API URL
        
#         Args:
#             data_source: 数据集名称
            
#         Returns:
#             API URL
#         """
#         # 精确匹配
#         if data_source in self.search_api_urls:
#             return self.search_api_urls[data_source]
        
#         # 部分匹配（处理路径中包含数据集名称的情况）
#         for dataset, url in self.search_api_urls.items():
#             if dataset in data_source or data_source in dataset:
#                 print(f"  部分匹配: {data_source} -> {dataset}")
#                 return url
        
#         # 未找到
#         raise ValueError(f"Port configuration not found for data source [{data_source}], available data sources: {list(self.search_api_urls.keys())}")

#     def get_dataset_ports(self) -> Dict[str, int]:
#         """
#         从配置文件读取数据集端口映射
        
#         Returns:
#             {dataset_name: port} 字典
#         """
#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
#         config_path = os.path.join(project_root, "config.json")
        
#         try:
#             with open(config_path, "r") as f:
#                 config = json.load(f)
                
#                 # 读取 dataset_ports
#                 if "dataset_ports" in config and config["dataset_ports"]:
#                     print(f"✓ 从 config.json 读取到端口映射: {config['dataset_ports']}")
#                     return config["dataset_ports"]
                
#                 return {}
                
#         except (FileNotFoundError, json.JSONDecodeError) as e:
#             print(f"⚠ 无法读取 config.json: {e}")
#             return {}

"""
Search tool implementation for simulating internet searches
"""

import time
import random
import os
from typing import Dict, List, Any, Optional

from agent.tool.tool_base import Tool

import faiss
from FlagEmbedding import FlagAutoModel
import json
import requests


class SearchTool(Tool):
    """
    Tool for simulating internet searches using the NeuML/txtai-wikipedia model
    """
    
    def __init__(self, config_name: Optional[str] = None):
        """
        Initialize the search tool
        
        Args:
            config_name: Config file name, e.g., "config_NQ_hypergraph_scale1.json"
                        If None, will try to read from environment variables or use default config
        """
        name = "search"
        description = "Search for information on the internet using Wikipedia as a knowledge source."
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
            },
            "required": ["query"]
        }
        
        super().__init__(name, description, parameters)
        
        # Determine config file
        self.config_name = config_name or self._get_config_name_from_env()
        
        # Read config
        self.config = self._load_config()
        self.dataset_ports = self.config.get("dataset_ports", {})
        
        # Build API URLs
        self.search_api_urls = {}
        for dataset, port in self.dataset_ports.items():
            self.search_api_urls[dataset] = f"http://localhost:{port}/search"
        
        self._print_init_info()

    def _get_config_name_from_env(self) -> str:
        """Get config file name from environment variables"""
        # Try to read from environment variables
        dataset = os.getenv("DATASET", "2WikiMultiHopQA")
        graphrag = os.getenv("GRAPHRAG", "hypergraph")
        node_scale = os.getenv("NODE_SCALE", "1")
        
        config_name = f"config_{dataset}_{graphrag}_scale{node_scale}.json"
        print(f"Building config file name from environment variables: {config_name}")
        
        return config_name

    def _load_config(self) -> Dict:
        """Load config file"""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        config_path = os.path.join(project_root, self.config_name)
        
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                print(f"✓ Successfully loaded config file: {self.config_name}")
                return config
        except FileNotFoundError:
            print(f"⚠ Config file does not exist: {config_path}")
            print(f"  Trying to use default config config.json")
            
            # Try to load default config
            default_config_path = os.path.join(project_root, "config.json")
            try:
                with open(default_config_path, "r") as f:
                    config = json.load(f)
                    print(f"✓ Successfully loaded default config: config.json")
                    return config
            except Exception as e:
                print(f"✗ Cannot load config file: {e}")
                return {}
        except json.JSONDecodeError as e:
            print(f"✗ Config file format error: {e}")
            return {}

    def _print_init_info(self):
        """Print initialization information"""
        print(f"\n{'='*60}")
        print(f"SearchTool Initialization")
        print(f"{'='*60}")
        print(f"Config file: {self.config_name}")
        print(f"Dataset: {self.config.get('dataset', 'N/A')}")
        print(f"GraphRAG: {self.config.get('graphrag', 'N/A')}")
        print(f"Node scale: {self.config.get('node_scale', 'N/A')}")
        print(f"Dataset port mapping:")
        for dataset, port in self.dataset_ports.items():
            print(f"  {dataset:20s}: port {port} -> {self.search_api_urls[dataset]}")
        print(f"{'='*60}\n")

    def execute(self, args: Dict) -> str:
        """
        Execute search query
        
        Args:
            args: Tool parameters, containing:
                - "query": search query string
            
        Returns:
            Formatted search results
        """
        pass
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """
        Execute search queries in batch, automatically route to different ports based on data_source
        
        Args:
            args_list: Parameter list, each element contains:
                - "query": Query string
                - "data_source": Dataset name (required)
            
        Returns:
            List of search results
        """
        queries = [x["query"] for x in args_list]
        data_sources = [x["data_source"] for x in args_list]
        
        # Group by data source, call different APIs
        return self._batch_call_by_datasource(queries, data_sources)

    def _batch_call_by_datasource(self, queries: List[str], data_sources: List[str]) -> List[str]:
        """
        Group by data source and call different API ports
        
        Args:
            queries: Query list
            data_sources: Corresponding data source list
            
        Returns:
            Result list (maintains original order)
        """
        # Group by data source
        source_groups = {}  # {data_source: [(index, query), ...]}
        
        for idx, (query, ds) in enumerate(zip(queries, data_sources)):
            if ds not in source_groups:
                source_groups[ds] = []
            source_groups[ds].append((idx, query))
        
        # Initialize result list
        results = [None] * len(queries)
        
        # Call corresponding API for each data source separately
        for ds_key, group in source_groups.items():
            indices = [item[0] for item in group]
            ds_queries = [item[1] for item in group]
            
            # Get corresponding API URL
            api_url = self.get_api_url(ds_key)
            
            print(f"  Batch call [{ds_key}]: {len(ds_queries)} queries -> {api_url}")
            
            try:
                response = requests.post(api_url, json={"queries": ds_queries})
                response.raise_for_status()
                ds_results = response.json()
                
                # Put results back to original positions
                for result, orig_idx in zip(ds_results, indices):
                    results[orig_idx] = result
                    
            except Exception as e:
                print(f"⚠ Failed to call [{ds_key}] API: {e}")
                # Fill error information for failed queries
                for orig_idx in indices:
                    results[orig_idx] = f"Error: {str(e)}"
        
        return results
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # valid tool call
        if "results" in result:
            return 0.0
        else:
            return 0.0

    def get_api_url(self, data_source: str) -> str:
        """
        Get corresponding API URL based on data source
        
        Args:
            data_source: Dataset name
            
        Returns:
            API URL
        """
        # Exact match
        if data_source in self.search_api_urls:
            return self.search_api_urls[data_source]
        
        # Partial match (handle cases where dataset name is included in path)
        for dataset, url in self.search_api_urls.items():
            if dataset in data_source or data_source in dataset:
                print(f"  Partial match: {data_source} -> {dataset}")
                return url
        
        # Not found
        raise ValueError(f"Port configuration not found for data source [{data_source}], available data sources: {list(self.search_api_urls.keys())}")