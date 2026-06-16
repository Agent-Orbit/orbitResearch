from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

from tools.llm import llmHandler
from tools import arxivSearch, githubSearch, webSearch, wikiSearch, ytSearch


class ResearchState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]  # agent scratchpad
    tool_outputs: list                        # raw results + sources
    sources_used: list                        # tracked tools that fired
    summary: str                              # final synthesized output
    model_history: list                       # compressed history for LLM
    grade: str                                # "pass" | "retry" | "fail"
    metadata: dict                            # timestamps, retry count, etc.


class CoreResearch:

    def __init__(self, model_name):
        self.llm_handler = llmHandler(model_name)
        self.llm = self.llm_handler.llm
        self.agent = self.makeAgent()
        self.compileGraph()

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
            system_prompt="You are a research assistant. Always cite which tool produced each result.",
            middleware=[
                SummarizationMiddleware(model=self.llm, trigger=[("messages", 10)]),
                ToolCallLimitMiddleware(run_limit=10),
            ]
        )

    def agent_node(self, state: ResearchState):

        result = self.agent.invoke({"messages": state["messages"]})
        sources = [m.name for m in result["messages"] if hasattr(m, "name")]

        return {
            "messages": result["messages"],
            "sources_used": sources,
            "model_history": result["messages"],
        }

    def grade_results(self, state: ResearchState):

        last = state["messages"][-1].content

        if not last or "I couldn't find" in last or len(last) < 100:

            return {"grade": "retry"}
        
        return {"grade": "pass"}

    def synthesize(self, state: ResearchState):

        content = state["messages"][-1].content
        sources = ", ".join(state["sources_used"]) if state["sources_used"] else "unknown"
        prompt = f"Summarize the following research findings. Sources used: {sources}\n\n{content}"
        summary = self.llm_handler.queryLLM(prompt)

        return {"summary": summary}

    def retry_query(self, state: ResearchState):

        retries = state.get("metadata", {}).get("retries", 0)

        return {
            "messages": [HumanMessage(content=f"Try again with a broader search: {state['query']}")],
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
            "model_history": [],
            "grade": "",
            "metadata": {"retries": 0}
        }

        result = self.graph.invoke(initial_state)

        return {
            "summary": result["summary"],
            "sources_used": result["sources_used"],
            "metadata": result["metadata"]
        }