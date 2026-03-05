# #!/usr/bin/env python
# """
# 这个脚本用于更新端口配置
# 使用方法: python update_port_config.py --port 8002
# """

# import json
# import os
# import argparse

# def update_port(port):
#     """更新配置文件中的端口设置"""
#     config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    
#     # 读取现有配置
#     try:
#         with open(config_path, "r") as f:
#             config = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError):
#         config = {}
    
#     # 更新端口
#     config["search_api_port"] = port
    
#     # 写回配置文件
#     with open(config_path, "w") as f:
#         json.dump(config, f, indent=4)
        
#     print(f"已更新API端口配置为: {port}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="更新搜索API端口配置")
#     parser.add_argument("--port", type=int, default=8002, help="API服务端口号 (默认: 8002)")
    
#     args = parser.parse_args()
#     update_port(args.port)
# #可用
# #!/usr/bin/env python
# """
# 这个脚本用于更新端口配置
# 支持单数据集和多数据集模式，统一使用 dataset_ports 格式

# 使用方法: 
#   单数据集: python update_port_config.py --port 8002 --dataset NQ
#   多数据集: python update_port_config.py --port 8002 --dataset NQ-HotpotQA
# """

# import json
# import os
# import argparse
# from typing import List, Dict
# import re


# def parse_dataset_names(dataset: str) -> List[str]:
#     """
#     解析数据集名称，支持混合数据集格式
    
#     Args:
#         dataset: 数据集名称，如 "NQ" 或 "NQ-HotpotQA" 或 "NQ+HotpotQA"
        
#     Returns:
#         数据集名称列表
#     """
#     if not dataset:
#         return []
    
#     # 使用 - 或 + 或 , 作为分隔符
#     datasets = re.split(r'[-+,]', dataset)
#     # 去除空格
#     return [ds.strip() for ds in datasets if ds.strip()]


# def update_port_config(port: int, dataset: str):
#     """
#     更新配置文件中的端口设置
#     统一使用 dataset_ports 格式，多数据集时递增端口
    
#     Args:
#         port: 起始端口号
#         dataset: 数据集名称，支持单个或混合数据集（用-或+分隔）
#     """
#     config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    
#     # 读取现有配置
#     try:
#         with open(config_path, "r") as f:
#             config = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError):
#         config = {}
    
#     # 解析数据集名称
#     dataset_list = parse_dataset_names(dataset)
    
#     if not dataset_list:
#         raise ValueError("必须指定数据集名称")
    
#     # 构建 dataset_ports 配置
#     config["dataset_ports"] = {}
#     current_port = port
    
#     for ds in dataset_list:
#         config["dataset_ports"][ds] = current_port
#         current_port += 1
    
#     # 写回配置文件
#     with open(config_path, "w") as f:
#         json.dump(config, f, indent=4)
    
#     # 打印配置信息
#     print(f"\n{'='*60}")
#     print(f"端口配置已更新")
#     print(f"{'='*60}")
    
#     if len(dataset_list) == 1:
#         print(f"模式: 单数据集")
#         print(f"数据集: {dataset_list[0]}")
#         print(f"端口: {port}")
#     else:
#         print(f"模式: 多数据集")
#         print(f"数据集组合: {dataset}")
#         print(f"端口映射:")
#         for ds, p in config["dataset_ports"].items():
#             print(f"  {ds:20s}: {p}")
    
#     print(f"{'='*60}")
#     print(f"\n配置已保存到: {config_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="更新搜索API端口配置（统一使用 dataset_ports 格式）",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# 使用示例:
#   1. 单数据集:
#      python update_port_config.py --port 8002 --dataset NQ
#      生成: {"dataset_ports": {"NQ": 8002}}
     
#   2. 两个数据集 (NQ-HotpotQA):
#      python update_port_config.py --port 8002 --dataset NQ-HotpotQA
#      生成: {"dataset_ports": {"NQ": 8002, "HotpotQA": 8003}}
     
#   3. 三个数据集:
#      python update_port_config.py --port 8002 --dataset NQ-HotpotQA-2WikiMultiHopQA
#      生成: {"dataset_ports": {"NQ": 8002, "HotpotQA": 8003, "2WikiMultiHopQA": 8004}}
#         """
#     )
    
#     parser.add_argument(
#         "--port", 
#         type=int, 
#         required=True,
#         help="API服务起始端口号"
#     )
#     parser.add_argument(
#         "--dataset", 
#         type=str, 
#         required=True,
#         help="数据集名称，支持单个或混合数据集（用-或+分隔），如: NQ 或 NQ-HotpotQA"
#     )
    
#     args = parser.parse_args()
#     update_port_config(args.port, args.dataset)
#!/usr/bin/env python
# """
# 这个脚本用于更新端口配置
# 支持单数据集和多数据集模式，统一使用 dataset_ports 格式

# 使用方法: 
#   单数据集: python update_port_config.py --port 8002 --dataset NQ
#   多数据集: python update_port_config.py --port 8002 --dataset NQ-HotpotQA
#   仅更新默认端口: python update_port_config.py --port 8002
# """

# import json
# import os
# import argparse
# from typing import List, Dict


# def parse_dataset_names(dataset: str) -> List[str]:
#     """
#     解析数据集名称，支持混合数据集格式
    
#     Args:
#         dataset: 数据集名称，如 "NQ" 或 "NQ-HotpotQA" 或 "NQ+HotpotQA"
        
#     Returns:
#         数据集名称列表
#     """
#     if not dataset:
#         return []
    
#     # 使用 - 或 + 或 , 作为分隔符
#     import re
#     datasets = re.split(r'[-+,]', dataset)
#     # 去除空格
#     return [ds.strip() for ds in datasets if ds.strip()]


# def update_port_config(port: int, dataset: str = None):
#     """
#     更新配置文件中的端口设置
#     统一使用 dataset_ports 格式
    
#     Args:
#         port: 起始端口号
#         dataset: 数据集名称，支持单个或混合数据集（用-或+分隔）
#     """
#     config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    
#     # 读取现有配置
#     try:
#         with open(config_path, "r") as f:
#             config = json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError):
#         config = {}
    
#     # 确保有 dataset_ports 字段
#     if "dataset_ports" not in config:
#         config["dataset_ports"] = {}
    
#     # 解析数据集名称
#     dataset_list = parse_dataset_names(dataset) if dataset else []
    
#     if not dataset_list:
#         # 没有指定数据集，只更新默认端口
#         config["search_api_port"] = port
#         print(f"✓ 已更新默认API端口为: {port}")
#         print(f"  注意: 未指定数据集，仅更新默认端口")
        
#     elif len(dataset_list) == 1:
#         # 单个数据集 - 也使用 dataset_ports 格式
#         single_dataset = dataset_list[0]
        
#         # 清空旧的 dataset_ports，设置新的单数据集配置
#         config["dataset_ports"] = {
#             single_dataset: port
#         }
#         config["search_api_port"] = port
        
#         print(f"✓ 已更新单数据集配置:")
#         print(f"  数据集: {single_dataset}")
#         print(f"  端口: {port}")
#         print(f"  配置格式: dataset_ports")
        
#     else:
#         # 多数据集模式 - 递增分配端口
#         config["dataset_ports"] = {}
#         current_port = port
#         port_mapping = {}
        
#         for ds in dataset_list:
#             config["dataset_ports"][ds] = current_port
#             port_mapping[ds] = current_port
#             current_port += 1
        
#         # 设置默认端口为第一个数据集的端口
#         config["search_api_port"] = port
        
#         print(f"✓ 已更新多数据集配置:")
#         print(f"  数据集组合: {dataset}")
#         print(f"  数据集数量: {len(dataset_list)}")
#         print(f"  端口映射 (dataset_ports):")
#         for ds, p in port_mapping.items():
#             print(f"    {ds:20s}: {p}")
    
#     # 写回配置文件
#     with open(config_path, "w") as f:
#         json.dump(config, f, indent=4)
    
#     print(f"\n✓ 配置已保存到: {config_path}")
#     print(f"\n当前 config.json 内容:")
#     print(json.dumps(config, indent=4))




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="更新搜索API端口配置（统一使用 dataset_ports 格式）",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# 使用示例:
#   1. 单数据集模式:
#      python update_port_config.py --port 8002 --dataset NQ
#      生成配置:
#      {
#          "dataset_ports": {
#              "NQ": 8002
#          },
#          "search_api_port": 8002
#      }
     
#   2. 多数据集模式 (混合训练 NQ-HotpotQA):
#      python update_port_config.py --port 8002 --dataset NQ-HotpotQA
#      生成配置:
#      {
#          "dataset_ports": {
#              "NQ": 8002,
#              "HotpotQA": 8003
#          },
#          "search_api_port": 8002
#      }
     
#   3. 三个数据集混合:
#      python update_port_config.py --port 8002 --dataset NQ-HotpotQA-2WikiMultiHopQA
#      生成配置:
#      {
#          "dataset_ports": {
#              "NQ": 8002,
#              "HotpotQA": 8003,
#              "2WikiMultiHopQA": 8004
#          },
#          "search_api_port": 8002
#      }
     
#   4. 查看当前配置:
#      python update_port_config.py --show
     
#   5. 生成独立的 port_mapping.json (可选):
#      python update_port_config.py --port 8002 --dataset NQ-HotpotQA --generate-mapping
#         """
#     )
    
#     parser.add_argument(
#         "--port", 
#         type=int, 
#         help="API服务起始端口号"
#     )
#     parser.add_argument(
#         "--dataset", 
#         type=str, 
#         help="数据集名称，支持单个或混合数据集（用-或+分隔），如: NQ 或 NQ-HotpotQA"
#     )
    
#     args = parser.parse_args()
    

#     update_port_config(args.port, args.dataset)

#!/usr/bin/env python
"""
This script is used to update port configuration
Supports single-dataset and multi-dataset modes, generates config files based on dataset, graphrag, node_scale

Usage: 
  python update_port_config.py --port 8002 --dataset NQ --graphrag hypergraph --node_scale 1
"""

import json
import os
import argparse
from typing import List, Dict, Optional
import re


def parse_dataset_names(dataset: str) -> List[str]:
    """
    Parse dataset names, supports mixed dataset format
    
    Args:
        dataset: Dataset name, e.g., "NQ" or "NQ-HotpotQA"
        
    Returns:
        List of dataset names
    """
    if not dataset:
        return []
    
    # Use - or + or , as separators
    datasets = re.split(r'[-+,]', dataset)
    # Remove whitespace
    return [ds.strip() for ds in datasets if ds.strip()]


def get_config_filename(dataset: str, graphrag: str, node_scale: int) -> str:
    """
    Generate config file name based on parameters
    
    Args:
        dataset: Dataset name
        graphrag: GraphRAG type
        node_scale: Node scale
        
    Returns:
        Config file name
    """
    return f"config_{dataset}_{graphrag}_scale{node_scale}.json"


def update_port_config(port: int, dataset: str, graphrag: str, node_scale: int, config_name: Optional[str] = None):
    """
    Update port configuration in config file
    
    Args:
        port: Starting port number
        dataset: Dataset name
        graphrag: GraphRAG type
        node_scale: Node scale
        config_name: Config file name (optional, auto-generated based on dataset_graphrag_scale if not provided)
    """
    # Use original naming when config_name is not provided
    if config_name:
        config_filename = config_name if config_name.endswith(".json") else f"{config_name}.json"
    else:
        config_filename = get_config_filename(dataset, graphrag, node_scale)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_filename)
    
    # Parse dataset names
    dataset_list = parse_dataset_names(dataset)
    
    if not dataset_list:
        raise ValueError("Dataset name must be specified")
    
    # Read existing config (if exists)
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}
    
    # Build dataset_ports configuration
    config["dataset_ports"] = {}
    current_port = port
    
    for ds in dataset_list:
        config["dataset_ports"][ds] = current_port
        current_port += 1
    
    # Add metadata
    config["dataset"] = dataset
    config["graphrag"] = graphrag
    config["node_scale"] = node_scale
    
    # Write back to config file
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Print configuration info
    print(f"\n{'='*60}")
    print(f"Port configuration updated")
    print(f"{'='*60}")
    print(f"Config file: {config_filename}")
    print(f"Dataset: {dataset}")
    print(f"GraphRAG: {graphrag}")
    print(f"Node scale: {node_scale}")
    
    if len(dataset_list) == 1:
        print(f"Mode: Single dataset")
        print(f"Port: {port}")
    else:
        print(f"Mode: Multi-dataset")
        print(f"Port mapping:")
        for ds, p in config["dataset_ports"].items():
            print(f"  {ds:20s}: {p}")
    
    print(f"{'='*60}")
    print(f"\nConfig saved to: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update search API port configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  1. Without specifying config_name (auto-named as config_{dataset}_{graphrag}_scale{node_scale}.json):
     python update_port_config.py --port 8002 --dataset NQ --graphrag hypergraphrag --node_scale 1
     
  2. With config_name specified:
     python update_port_config.py --port 8002 --dataset NQ --graphrag hypergraphrag --node_scale 1 --config_name config_NQ_hypergraphrag_scale1.json
     
  3. Multi-dataset:
     python update_port_config.py --port 8002 --dataset NQ-HotpotQA --graphrag lightrag --node_scale 1000
        """
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        required=True,
        help="API service starting port number"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--graphrag",
        type=str,
        required=True,
        choices=["hypergraphrag", "lightrag", "hipporag", "linearrag", "raptor", "graphrag"],
        help="GraphRAG type"
    )
    parser.add_argument(
        "--node_scale",
        type=int,
        required=True,
        help="Node scale"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Config file name (optional; auto-generated as config_{dataset}_{graphrag}_scale{node_scale}.json if not provided)"
    )
    
    args = parser.parse_args()
    update_port_config(args.port, args.dataset, args.graphrag, args.node_scale, args.config_name)