from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from research_core.llm import llmHandler
from tools import arxivSearch, githubSearch, webSearch, wikiSearch, ytSearch
from langchain_core.messages import SystemMessage
from utils.prompts_loader import get as get_prompt


class ResearchState(TypedDict):

    query: str
    messages: Annotated[list, add_messages]  # agent scratchpad
    tool_outputs: list                        # raw results + sources
    sources_used: list                        # tracked tools that fired
    summary: str                              # final synthesized output
    model_history: list                       # compressed history for LLM
    grade: str                                # "pass" | "retry" | "fail"
    metadata: dict                           # timestamps, retry count, etc.


class CoreResearch:

    def __init__(self, model_name):
        self.llm_handler = llmHandler(model_name)
        self.llm = self.llm_handler.llm
        self.agent = self.makeAgent()
        self.compileGraph()
        self.model_history = [] 
        

    def makeAgent(self):

        return create_agent(
            model=self.llm,
            tools=[
                arxivSearch.arxivSearch,
                githubSearch.githubSearchTool,
                webSearch.search,
                webSearch.extractWeb,
                wikiSearch.wikipediaSearch,
                ytSearch.youtubeSearch,        
                ytSearch.youtubeVideoDetails,
            ],
            system_prompt=get_prompt("agent_system"),
            middleware=[
                SummarizationMiddleware(model=self.llm, trigger=[("messages", 10)]),
                ToolCallLimitMiddleware(run_limit=10),
            ]
        )

    def agent_node(self, state: ResearchState):

        result = self.agent.invoke({"messages": state["model_history"] + state["messages"]})
        sources = [m.name for m in result["messages"] if hasattr(m, "name") and m.name is not None]

        updated_history = state.get("model_history", []) + result["messages"]
        compressed = self._maybe_summarize(updated_history)

        return {
            "messages": result["messages"],
            "sources_used": sources,
            "model_history": compressed
        }

    def grade_results(self, state: ResearchState):

        last = state["messages"][-1].content

        if not last or "I couldn't find" in last or len(last) < 100:

            return {"grade": "retry"}
        
        return {"grade": "pass"}

    def synthesize(self, state: ResearchState):

        content = state["messages"][-1].content
        sources = ", ".join(state["sources_used"]) if state["sources_used"] else "unknown"
        prompt = get_prompt("synthesize", query=state["query"], sources=sources, content=content)
        summary = self.llm_handler.queryLLM(prompt)

        return {"summary": summary}

    def retry_query(self, state: ResearchState):

        retries  = state.get("metadata", {}).get("retries", 0)
        original = state["query"]

        revised_query = self.llm_handler.queryLLM(
            get_prompt("rephrase_query", query=original)
        ).strip().strip('"')

        return {
            "messages": [HumanMessage(content=get_prompt(
                "retry",
                query=original,
                attempt=retries + 1,
                revised_query=revised_query,
            ))],

            "metadata": {**state.get("metadata", {}), "retries": retries + 1}
        }

    def route_grade(self, state: ResearchState):

        if state["grade"] == "retry" and state.get("metadata", {}).get("retries", 0) < 1:

            return "retry"
        
        return "synthesize"

    def compileGraph(self):

        graph = StateGraph(ResearchState)
        graph.add_node("agent", self.agent_node)
        graph.add_node("grade_results", self.grade_results)
        graph.add_node("synthesize", self.synthesize)
        graph.add_node("retry", self.retry_query)

        graph.add_edge(START, "agent")
        graph.add_edge("agent", "grade_results")
        graph.add_conditional_edges("grade_results", self.route_grade, {
            "retry": "agent",
            "synthesize": "synthesize"
        })
        graph.add_edge("synthesize", END)

        self.graph = graph.compile()
    
    def run(self, query: str) -> dict:

        initial_state = {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "tool_outputs": [],
            "sources_used": [],
            "summary": "",
            "model_history": self.model_history,
            "grade": "",
            "metadata": {"retries": 0}
        }

        result = self.graph.invoke(initial_state)
        self.model_history = result["model_history"]

        return {
            "summary": result["summary"],
            "sources_used": result["sources_used"],
            "metadata": result["metadata"],
        }
    
    def _maybe_summarize(self, messages: list) -> list:
        """Summarize history once it exceeds 10 messages, keep last 4 as-is."""

        THRESHOLD = 10
        KEEP_RECENT = 4

        if len(messages) <= THRESHOLD:

            return messages                       

        to_summarize = messages[:-KEEP_RECENT]    
        recent = messages[-KEEP_RECENT:]    

        combined = "\n".join(
            m.content for m in to_summarize if hasattr(m, "content") and m.content
        )
        summary_text = self.llm_handler.queryLLM(
            f"Summarize these research findings concisely:\n\n{combined}"
        )

        
        return [SystemMessage(content=f"[Prior context summary]: {summary_text}")] + recent