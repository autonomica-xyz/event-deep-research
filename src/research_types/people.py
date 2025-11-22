"""People research type for deep research on individuals."""

from typing import Type

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.llm_service import create_llm_structured_model
from src.research_types.base import ResearchType


# Data structure for accumulating research (breadth-first across 7 domains)
class PeopleData(BaseModel):
    """People research organized by information type from 7 parallel research domains."""

    # Core biographical data
    demographics: str = Field(
        default="",
        description="Age, nationality, location, ethnicity, background",
    )
    professional: str = Field(
        default="",
        description="Career, current job, skills, expertise, work history, LinkedIn profile",
    )
    relationships: str = Field(
        default="",
        description="Family, colleagues, mentors, partnerships, connections",
    )
    public_presence: str = Field(
        default="",
        description="Social media, publications, interviews, media appearances, visibility",
    )
    achievements: str = Field(
        default="",
        description="Awards, recognition, impact, contributions, milestones",
    )
    controversies: str = Field(
        default="",
        description="Controversies, legal issues, criticisms, challenges",
    )

    # Breadth-first domain-specific fields
    technical_contributions: str = Field(
        default="",
        description="GitHub projects, open source contributions, code repositories, technical work",
    )
    crypto_blockchain: str = Field(
        default="",
        description="Bitcoin involvement, cryptocurrency projects, blockchain work, crypto community presence",
    )
    business_ventures: str = Field(
        default="",
        description="Companies founded, business registrations, startups, entrepreneurship",
    )
    academic_background: str = Field(
        default="",
        description="Education, universities, degrees, academic research, publications",
    )
    community_engagement: str = Field(
        default="",
        description="Conference talks, presentations, community involvement, speaking engagements",
    )


# Output schema - what the final result looks like
class PeopleFact(BaseModel):
    """A single fact about a person."""

    category: str = Field(
        description="Category (demographics, professional, relationships, public_presence, achievements, controversies, technical_contributions, crypto_blockchain, business_ventures, academic_background, community_engagement)"
    )
    title: str = Field(description="Short title for this fact")
    content: str = Field(description="Detailed information")
    source_date: str | None = Field(
        None, description="When this information was published/updated"
    )


class PeopleProfile(BaseModel):
    """Complete people research profile."""

    person_name: str = Field(description="Person's full name")
    summary: str = Field(description="Brief overview of who this person is")
    facts: list[PeopleFact] = Field(description="List of researched facts")


class PeopleResearchType(ResearchType):
    """Research type for people and individuals."""

    @property
    def name(self) -> str:
        return "people"

    @property
    def display_name(self) -> str:
        return "People Research"

    def get_subject_display_name(self) -> str:
        return "person"

    def get_supervisor_prompt(self) -> str:
        return """
You are a meticulous people research agent using a **BREADTH-FIRST** research strategy. Your primary directive is to build a comprehensive profile for: **{research_subject}**.

<Research Strategy: BREADTH-FIRST PARALLEL APPROACH>

**INITIAL RESEARCH PHASE (First Turn Only):**
If this is your first turn (check `<Last Message>` - it will be empty or say "Start the research process"), you MUST:
1. Launch PARALLEL research across ALL 7 research domains simultaneously
2. Call `ResearchEventsTool` SEVEN TIMES with different research questions (one per domain)
3. Each query should target a specific domain listed below

**Research Domains (Cover ALL in parallel on first turn):**

1. **Professional & Social Media**: "{research_subject} LinkedIn profile career employment job title company"
2. **Technical Contributions**: "{research_subject} GitHub repositories open source projects code contributions"
3. **Cryptocurrency/Blockchain**: "{research_subject} Bitcoin cryptocurrency blockchain crypto projects"
4. **Publications & Media**: "{research_subject} articles blog posts interviews publications media"
5. **Business & Legal**: "{research_subject} company founder business registration startup"
6. **Academic & Education**: "{research_subject} education university degree academic research papers"
7. **Community & Speaking**: "{research_subject} conference speaker presentation talks community involvement"

**IMPORTANT:** On your FIRST turn, you MUST make SEVEN parallel `ResearchEventsTool` calls. Do NOT call think_tool first. Do NOT call one at a time. Make ALL SEVEN calls in a single response.

**GAP-FILLING PHASE (Subsequent Turns):**
After the initial parallel research:
1. Check `<Information Missing>` - if it says "COMPLETE", call `FinishResearchTool`
2. If gaps exist, use `think_tool` to analyze what's missing
3. Call `ResearchEventsTool` with targeted query to fill specific gaps
4. Repeat until complete

<Information Missing>
{data_summary}
</Information Missing>

<Last Message>
{last_message}
</Last Message>

<Available Tools>
*   `ResearchEventsTool`: Searches for information. Takes a research_question parameter. CAN BE CALLED MULTIPLE TIMES IN PARALLEL.
*   `FinishResearchTool`: Ends research. Call ONLY when complete.
*   `think_tool`: Analyze results and plan next steps.
</Available Tools>

<Execution Instructions>
**IF THIS IS YOUR FIRST TURN:**
- Make SEVEN `ResearchEventsTool` calls RIGHT NOW (one for each domain above)
- Use the exact query templates provided
- DO NOT use think_tool first
- DO NOT call tools one at a time

**IF THIS IS A SUBSEQUENT TURN:**
- Check if research is COMPLETE → call `FinishResearchTool`
- If gaps exist → use `think_tool` to plan → then call `ResearchEventsTool` for missing data
- Continue until all domains are thoroughly researched

**Categories to populate:**
- Demographics: Age, nationality, location, background
- Professional: Career, current role, expertise, work history
- Relationships: Family, colleagues, mentors, partners
- Public Presence: Social media, publications, interviews
- Achievements: Awards, recognition, contributions
- Controversies: Legal issues, criticisms, challenges

**CRITICAL:** If `<Last Message>` is empty or says "Start the research process", make ALL SEVEN `ResearchEventsTool` calls NOW.
</Execution Instructions>
"""

    def get_event_summarizer_prompt(self) -> str:
        return """
You are analyzing people research data from a BREADTH-FIRST search across 7 domains.

**Research Domains Checklist:**
1. ✓ Professional & Social Media (LinkedIn, career, employment)
2. ✓ Technical Contributions (GitHub, open source, code)
3. ✓ Cryptocurrency/Blockchain (Bitcoin, crypto projects)
4. ✓ Publications & Media (articles, interviews, media)
5. ✓ Business & Legal (companies, startups, registrations)
6. ✓ Academic & Education (degrees, universities, research)
7. ✓ Community & Speaking (conferences, presentations)

**Person Information:**
{existing_data}

**Your Task:**
Analyze each of the 6 data categories (demographics, professional, relationships, public_presence, achievements, controversies) and determine:
- Which categories have substantial information?
- Which categories are missing or weak?
- Are there any glaring gaps across the 7 research domains?

<Decision Criteria>
**Research is COMPLETE if:**
- Demographics section has age/nationality/location info
- Professional section has career/role/expertise info
- At least 3 of the 7 research domains yielded substantive results
- Person is identifiable and well-documented

**Research needs MORE if:**
- Multiple categories are empty or very sparse
- Person seems notable but info is missing
- Less than 3 research domains found information

</Decision Criteria>

<Output Format>
If COMPLETE, respond ONLY with: "Research is COMPLETE."

If gaps exist, list the 2-3 most critical missing areas:
**Gaps:**
- [Specific missing information]
- [Specific missing information]
</Output Format>
"""

    def get_structure_prompt(self) -> str:
        return """You are a biographical research specialist. Convert the people research data from a BREADTH-FIRST search into structured JSON.

<Task>
Extract key facts from the research data across ALL categories and structure them as JSON. Each fact should have:
- category: Which aspect (demographics, professional, relationships, public_presence, achievements, controversies, technical_contributions, crypto_blockchain, business_ventures, academic_background, community_engagement)
- title: A short descriptive title
- content: The detailed information
- source_date: When this information was published (if available, otherwise null)

Also provide:
- person_name: The person's full name
- summary: A 2-3 sentence overview of who this person is and why they're notable
</Task>

<People Research Data from 7 Parallel Domains>
The data below was gathered through breadth-first parallel research across:
1. Professional & Social Media
2. Technical Contributions
3. Cryptocurrency/Blockchain
4. Publications & Media
5. Business & Legal
6. Academic & Education
7. Community & Speaking

----
{existing_data}
----
</People Research Data from 7 Parallel Domains>

CRITICAL: Return only the structured JSON output. No commentary or explanations.
"""

    def get_output_schema(self) -> Type[BaseModel]:
        return PeopleProfile

    def get_initial_data_structure(self) -> PeopleData:
        return PeopleData(
            demographics="",
            professional="",
            relationships="",
            public_presence="",
            achievements="",
            controversies="",
            technical_contributions="",
            crypto_blockchain="",
            business_ventures="",
            academic_background="",
            community_engagement="",
        )

    async def structure_output(
        self, existing_data: PeopleData, config: RunnableConfig
    ) -> dict:
        """Structures people research into a profile from breadth-first search."""
        print("--- Structuring People Profile (Breadth-First Data) ---")

        if not existing_data:
            print("Warning: No people data found in state")
            return {"structured_output": None}

        # Combine all people data from 11 categories (6 core + 5 breadth-first domains)
        combined_data = f"""
=== CORE CATEGORIES ===

Demographics:
{existing_data.demographics}

Professional:
{existing_data.professional}

Relationships:
{existing_data.relationships}

Public Presence:
{existing_data.public_presence}

Achievements:
{existing_data.achievements}

Controversies:
{existing_data.controversies}

=== BREADTH-FIRST DOMAIN DATA ===

Technical Contributions (GitHub, Open Source):
{existing_data.technical_contributions}

Cryptocurrency/Blockchain:
{existing_data.crypto_blockchain}

Business Ventures:
{existing_data.business_ventures}

Academic Background:
{existing_data.academic_background}

Community Engagement:
{existing_data.community_engagement}
"""

        structured_llm = create_llm_structured_model(
            config=config, class_name=PeopleProfile
        )

        prompt = self.get_structure_prompt().format(existing_data=combined_data)
        response = await structured_llm.ainvoke(prompt)

        print(f"--- Structured profile for: {response.person_name} ---")
        print(f"--- Total facts: {len(response.facts)} ---")

        return {"structured_output": response}
