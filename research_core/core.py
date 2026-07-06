import time
from typing import TypedDict, Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from research_core.llm import llmHandler
from tools import arxivSearch, githubSearch, webSearch, wikiSearch, ytSearch
from utils.prompts_loader import get as get_prompt


class ResearchState(TypedDict):

    query: str
    messages: Annotated[list, add_messages]   # scratchpad for THIS run only
    tool_outputs: list                        # raw tool results, tagged by tool name
    sources_used: list                        # tools that fired so far this run
    summary: str                              # final synthesized output
    grade: str                                # "pass" | "retry" | "fail"
    metadata: dict                            # timestamps, retry count, etc.


class CoreResearch:

    MAX_RETRIES = 1
    RECURSION_LIMIT = 15

    # Cross-turn memory budget. Kept OUTSIDE the graph state (see run/run_stream)
    # so it can never get double-counted with the in-run "messages" scratchpad.
    HISTORY_MESSAGE_LIMIT = 12
    HISTORY_CHAR_LIMIT = 12000       # rough proxy for ~3k tokens (1 token ~ 4 chars)
    SUMMARY_INPUT_CHAR_CAP = 2000    # per-message cap fed into any summarizer prompt

    def __init__(self, model_name: str):

        self.model_name = model_name
        self.llm_handler = llmHandler(model_name)
        self.llm = self.llm_handler.llm

        # Long-term, compressed conversation memory. This is the ONLY place
        # cross-turn history lives - it is never mixed back into state["messages"]
        # more than once per run.
        self.model_history: list = []

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

            system_prompt=get_prompt("agent_system"),
            middleware=[

                SummarizationMiddleware(

                    model=self.llm,
                    trigger=[("tokens", 3000), ("messages", 10)],
                    keep=("messages", 4),
                ),

                ToolCallLimitMiddleware(run_limit=10),
            ],
        )

    

    def agent_node(self, state: ResearchState):

        try:

            result = self.agent.invoke({"messages": state["messages"]})

        except Exception as e:
            
            return {

                "messages": [AIMessage(content=f"I couldn't complete the research due to an error: {e}")],
                "sources_used": state.get("sources_used", []),
                "tool_outputs": state.get("tool_outputs", []),
                "grade": "fail",
            }

        new_messages = result["messages"]

        new_sources = [m.name for m in new_messages if getattr(m, "name", None)]
        new_tool_outputs = [

            {"tool": m.name, "content": str(m.content)[: self.SUMMARY_INPUT_CHAR_CAP]}
            for m in new_messages
            if isinstance(m, ToolMessage)

        ]

        return {

            "messages": new_messages,
            "sources_used": state.get("sources_used", []) + new_sources,
            "tool_outputs": state.get("tool_outputs", []) + new_tool_outputs,
        }

    @staticmethod
    def _last_ai_message(messages: list):

        for m in reversed(messages):

            if isinstance(m, AIMessage) and m.content:

                return m
            
        return messages[-1] if messages else None

    def grade_results(self, state: ResearchState):

        if state.get("grade") == "fail":

            return {"grade": "fail"}

        last = self._last_ai_message(state["messages"])
        content = last.content if last else ""
        content = content if isinstance(content, str) else str(content)

        if not content or "I couldn't find" in content or len(content) < 100:

            return {"grade": "retry"}

        return {"grade": "pass"}

    def synthesize(self, state: ResearchState):

        last = self._last_ai_message(state["messages"])
        content = last.content if last else "Something went wrong before a research answer could be produced."
        content = content if isinstance(content, str) else str(content)

        if state.get("grade") == "fail":

            return {"summary": content}

        sources = ", ".join(sorted(set(state["sources_used"]))) if state["sources_used"] else "unknown"
        prompt = get_prompt("synthesize", query=state["query"], sources=sources, content=content)

        try:

            summary = self.llm_handler.queryLLM(prompt)

        except Exception as e:

            summary = f"{content}\n\n_(Note: the final write-up step failed: {e}. Showing the raw findings instead.)_"

        return {"summary": summary}

    def retry_query(self, state: ResearchState):

        retries = state.get("metadata", {}).get("retries", 0)
        original = state["query"]

        try:

            revised_query = self.llm_handler.queryLLM(
                get_prompt("rephrase_query", query=original)
            ).strip().strip('"')

        except Exception:

            revised_query = original

        return {
            "messages": [HumanMessage(content=get_prompt(
                "retry",
                query=original,
                attempt=retries + 1,
                revised_query=revised_query,
            ))],
            "metadata": {**state.get("metadata", {}), "retries": retries + 1},
        }

    def route_grade(self, state: ResearchState):

        if state["grade"] == "fail":

            return "synthesize"

        if state["grade"] == "retry" and state.get("metadata", {}).get("retries", 0) < self.MAX_RETRIES:

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
            "synthesize": "synthesize",
        })
        graph.add_edge("synthesize", END)

        self.graph = graph.compile()

    

    def _build_initial_state(self, query: str) -> ResearchState:

        # model_history (long-term, already-compressed memory) is prepended
        # exactly once here. From this point on, state["messages"] is the single
        # source of truth for the run - it is never re-concatenated with
        # model_history again, which is what caused the old duplication bug.

        return {

            "query": query,
            "messages": self.model_history + [HumanMessage(content=query)],
            "tool_outputs": [],
            "sources_used": [],
            "summary": "",
            "grade": "",
            "metadata": {"retries": 0, "started_at": time.time()},
        }

    def run(self, query: str) -> dict:

        initial_state = self._build_initial_state(query)
        result = self.graph.invoke(initial_state, config={"recursion_limit": self.RECURSION_LIMIT})

        self.model_history = self._maybe_summarize(result["messages"])

        metadata = result["metadata"]
        metadata["elapsed_seconds"] = round(time.time() - metadata.get("started_at", time.time()), 2)

        return {

            "summary": result["summary"],
            "sources_used": sorted(set(result["sources_used"])),
            "tool_outputs": result["tool_outputs"],
            "metadata": metadata,
        }

    def run_stream(self, query: str):
        """
        Generator version of run(), for UIs that want progress updates while the
        graph is executing. Yields short status strings, then yields the final
        result dict (same shape as run()'s return value) as the LAST item.
        """

        initial_state = self._build_initial_state(query)
        prev_state = initial_state
        final_state = initial_state

        for state in self.graph.stream(
            initial_state,
            config={"recursion_limit": self.RECURSION_LIMIT},
            stream_mode="values",
        ):
            
            if len(state["messages"]) > len(prev_state["messages"]) and not state.get("grade"):

                yield "🔎 Calling tools and reading results..."

            elif state.get("grade") and state["grade"] != prev_state.get("grade"):

                if state["grade"] == "retry":

                    yield "📉 First pass was thin, refining the query..."

                elif state["grade"] == "fail":
                    yield "⚠️ Hit an error, wrapping up with what we have..."

                else:
                    yield "📊 Findings look solid, moving to synthesis..."

            elif state.get("metadata", {}).get("retries", 0) > prev_state.get("metadata", {}).get("retries", 0):

                yield "🔁 Retrying with a revised query..."

            elif state.get("summary") and state["summary"] != prev_state.get("summary"):

                yield "✍️ Writing the final answer..."

            prev_state = state
            final_state = state

        self.model_history = self._maybe_summarize(final_state["messages"])

        metadata = final_state["metadata"]
        metadata["elapsed_seconds"] = round(time.time() - metadata.get("started_at", time.time()), 2)

        yield {
            "summary": final_state["summary"],
            "sources_used": sorted(set(final_state["sources_used"])),
            "tool_outputs": final_state["tool_outputs"],
            "metadata": metadata,
        }

    def reset(self):
        """Clear cross-turn memory - wire this up to a 'New conversation' button."""
        self.model_history = []

    

    def _maybe_summarize(self, messages: list) -> list:
        """
        Compress cross-turn memory once it passes a message-count OR a rough
        character/token budget, whichever comes first. Always keeps the most
        recent messages untouched for immediate context.
        """

        over_count = len(messages) > self.HISTORY_MESSAGE_LIMIT
        approx_chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
        over_chars = approx_chars > self.HISTORY_CHAR_LIMIT

        if not (over_count or over_chars):

            return messages

        keep_recent = 4
        to_summarize = messages[:-keep_recent] if len(messages) > keep_recent else messages
        recent = messages[-keep_recent:] if len(messages) > keep_recent else []

        combined = "\n".join(
            str(m.content)[: self.SUMMARY_INPUT_CHAR_CAP]
            for m in to_summarize
            if getattr(m, "content", None)
        )

        if not combined:

            return recent

        try:
            summary_text = self.llm_handler.queryLLM(
                "Summarize these research findings concisely, keeping any concrete facts, "
                f"numbers, and sources:\n\n{combined}"
            )
        except Exception:
            
            return recent

        return [SystemMessage(content=f"[Prior context summary]: {summary_text}")] + recent