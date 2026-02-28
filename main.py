import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict, Annotated
from langchain_tavily import TavilySearch
import getpass

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

class MessagesState(TypedDict):
    OptimizedQuery: str
    messages: Annotated[list, "Conversation messages"]
    citations: list
    llm_calls: int

query = input("Enter your Query: ")

initialState = {
    "OptimizedQuery": "",
    "messages": [HumanMessage(content=query)],
    "citations": [],
    "llm_calls": 0
}

def researchQueryOptimizer(state: MessagesState):
    response = gemini_llm.invoke(
        [
            HumanMessage(
                content=f"""
                You are an advanced Research Query Generation Agent.Your task is to analyze a user’s question and generate a concise, high-quality set of optimized research queries that, if executed, would allow a researcher or system to produce a detailed and comprehensive answer.You do NOT answer the original question.You ONLY generate short, efficient, strategically designed research queries.Your output must help a downstream research system gather structured, complete, and relevant information.Core ObjectivesWhen given a user query:Identify the primary intentBreak it into core research dimensionsGenerate a minimal but sufficient set of queriesAvoid redundancyEnsure coverage of:Definitions or background (if needed)Key components or mechanismsComparisons (if relevant)Data/statistics (if relevant)Case studies or examples (if helpful)Recent developments (if time-sensitive)Expert opinions or debates (if analytical)Query Design RulesEach generated query must be:Short (1 sentence maximum)Search-optimizedSpecificNon-overlappingCapable of returning high-signal informationAvoid:Vague phrasingConversational toneYes/no questionsRedundant variants of the same queryOver-fragmentation into trivial sub-questionsOutput FormatUse the following structure:User Question<original question>Research Queries:<query><query><query><query>Use numbered lists only.Do not provide explanations.Do not provide reasoning.Do not answer the question.Do not add commentary.Query Structuring FrameworkWhen designing queries, follow this internal structure:Conceptual UnderstandingDefinitionsBackgroundHistorical context (if relevant)Mechanisms or ComponentsHow it worksKey elementsProcesses involvedEvidence or DataStatisticsEmpirical findingsMeasured impactComparisons or AlternativesVersus other approachesStrengths and weaknessesApplications or Case StudiesReal-world usageIndustry examplesGeographic differencesCurrent Trends (if applicable)Latest developmentsRegulatory updatesEmerging innovationsNot all categories are required. Select only what is relevant.ExamplesExample 1User Question:How does blockchain improve supply chain transparency?Research Queries:What is blockchain technology and how does it function?How is blockchain applied in supply chain management?How does blockchain improve traceability and transparency in supply chains?Case studies of blockchain implementation in global supply chainsLimitations and challenges of blockchain in supply chain systems , The Origianl questio is: 
                {state['messages'][-1].content}
                """
            )
        ]
    )
    return {"OptimizedQuery": response.content}


# def agenticInformationProcessing(state: MessagesState):
#     response = gemini_llm.invoke(
#         [
#             HumanMessage(
#                 content=f"""
#                 Provide expert research information about:
#                 {state['OptimizedQuery']}
#                 """
#             )
#         ]
#     )
#     return {"messages": state["messages"] + [response]}

def searchAPI(state: MessagesState):
    tool_search = TavilySearch(max_results=5)
    result = tool_search.invoke({"query": state["OptimizedQuery"]})
    search_results = result.get("results", [])
    return {"citations": search_results}


def ResearchAnalyser(state: MessagesState):
    citations_text = "\n".join(
        [f"{item['url']}\n{item['content']}" for item in state["citations"]]
    )
    response = gemini_llm.invoke(
        [
            HumanMessage(
                content=f"""
                You are an expert research analyst.Your task is to analyze research data collected from multiple websites and generate a comprehensive, evidence-based answer to the user’s question.You must carefully evaluate each website individually before constructing the final response.Each research entry contains:URLExtracted content from that URLYour Tasks1. Source-Level AnalysisFor each URL:Determine whether the content is relevant to the user’s questionExtract the most important claims, facts, or dataExplain briefly how this source contributes to answering the questionIgnore irrelevant or low-value contentYou must evaluate every source provided.2. Evidence-Based SynthesisAfter analyzing all sources:combine relevant insights into a structured, coherent responseRemove duplicationResolve contradictions if presentPrioritize high-quality, factual information3. Citation RulesInsert citation links inline immediately after the supporting sentenceUse the actual URL providedFormat citations like this:( Source: https://example.com)Do not group citations at the endDo not create a references sectionDo not fabricate URLsOnly cite URLs that were provided in the Research Data4. Output StructureYour final response must contain:Detailed AnswerStructured explanationClear paragraphsInline citationsSummaryConcise summary of the findingsNo new claimsStrict RequirementsDo not mention that you were given research dataDo not describe your reasoning processDo not explain your methodologyDo not invent factsDo not add information beyond what is supported in the provided sourcesIf sources conflict, present both perspectives with citationsIf insufficient information exists, state that clearlyQuality StandardYour answer must:Be analytical, not just descriptiveDemonstrate source evaluationClearly justify claims using citationsMaintain an expert, journalistic toneBe comprehensive but not repetitive

                User Question:{state['messages'][0].content}
                Research Data:{citations_text}
                """
            )
        ]
    )

    return {"messages": state["messages"] + [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("researchQueryOptimizer", researchQueryOptimizer)
workflow.add_node("searchAPI", searchAPI)
workflow.add_node("ResearchAnalyser", ResearchAnalyser)

workflow.add_edge(START, "researchQueryOptimizer")
workflow.add_edge("researchQueryOptimizer", "searchAPI")
workflow.add_edge("searchAPI", "ResearchAnalyser")
workflow.add_edge("ResearchAnalyser", END)

app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke(initialState)
    print("\n\nFINAL RESPONSE:\n")
    print(result["messages"][-1].content)