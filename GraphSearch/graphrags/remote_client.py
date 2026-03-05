import aiohttp
import json
import logging
from graphrags.base import GraphRAGBase

class RemoteGraphRAGClient(GraphRAGBase):
    def __init__(self, api_url: str, top_k: int = 5):
        # 调用父类 init，但传入 None 占位，因为我们不需要本地加载模型
        # 我们不需要 grag 对象，也不需要 QueryParam 类
        super().__init__(grag=None, QueryParam=None, grag_mode="remote", top_k=top_k)
        self.api_url = api_url
        self.retrieval_count = 0

    def reset_counts(self):
        self.retrieval_count = 0

    def init_graphrag(self, working_dir: str, EMBED_MODEL):
        # 远程模式下，Client 不需要初始化本地向量库
        pass

    def context_filter(self, context_data: str, filter_type: str) -> str:
        """
        适配 pipeline.py 的调用。
        由于 Server 端目前返回的是混合结果，这里暂时做透传。
        如果未来 Server 端分离了 '/search/semantic' 和 '/search/relational'，可以在这里通过逻辑区分。
        """
        # 可以在这里加简单的日志，观察数据流
        # logging.info(f"Filtering context for {filter_type} (Pass-through)")
        return context_data

    async def aquery_context(self, question: str):
        """
        重写父类方法，拦截对 self.grag.aquery 的调用，改为 HTTP 请求
        """
        return await self._call_remote_api(question)

    async def aquery_answer(self, question: str):
        """
        如果 pipeline 需要直接问答案（naive_grag_reasoning），也走这个接口
        """
        return await self._call_remote_api(question)

    async def _call_remote_api(self, question: str) -> str:
        """
        实际发送请求到 script_api_HypergraphRAG.py 启动的 Server
        """
        self.retrieval_count += 1
        async with aiohttp.ClientSession() as session:
            # Server 期望的格式是 {"queries": ["question"]}
            payload = {"queries": [question]}
            try:
                async with session.post(self.api_url, json=payload) as response:
                    if response.status == 200:
                        # Server 返回的是一个列表: [json_string_result1, json_string_result2...]
                        response_data = await response.json()
                        
                        if isinstance(response_data, list) and len(response_data) > 0:
                            # 提取第一个结果 (对应我们的单条 question)
                            raw_result = response_data[0]
                            
                            # Server 代码显示它做了一次 json.dumps({"results": ...})
                            # 所以这里可能需要解析一次 JSON 才能拿到纯文本，或者直接返回
                            try:
                                parsed_res = json.loads(raw_result)
                                # 假设我们只需要 'results' 字段里的文本内容
                                if isinstance(parsed_res, dict) and "results" in parsed_res:
                                     # 这里的 results 可能是 list，需要 join 成字符串供 pipeline 使用
                                    content = parsed_res["results"]
                                    if isinstance(content, list):
                                        return "\n".join([str(c) for c in content])
                                    return str(content)
                                return raw_result
                            except json.JSONDecodeError:
                                # 如果不是 JSON 字符串，直接返回
                                return raw_result
                        return ""
                    else:
                        logging.error(f"Remote RAG Server Error: {response.status} - {await response.text()}")
                        return "Error retrieving context from remote server."
            except Exception as e:
                logging.error(f"Connection failed to {self.api_url}: {e}")
                return f"Connection error: {e}"