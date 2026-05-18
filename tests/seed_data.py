"""
Synthetic article seed data for pipeline testing.
Three story clusters × three leans (left / center / right) = 9 articles total.
"""

TEST_QUERIES = {
    "ask": "What happened at the global climate summit and how did coverage differ?",
    "fact_check": "The AI regulation bill passed with bipartisan support in the Senate committee",
}

SEED_ARTICLES = [
    # ── Cluster 1: Global Climate Summit ──────────────────────────────────────
    {
        "url": "https://theguardian.com/test/climate-summit-left",
        "url_hash": "test_climate_left_001",
        "title": "World leaders agree historic climate deal as activists demand more ambition",
        "body": (
            "World leaders at the Global Climate Summit in Dubai signed a landmark agreement "
            "committing developed nations to cut carbon emissions by 45% before 2035. "
            "Environmental groups praised the targets but warned they fall short of the "
            "science-based 1.5°C pathway. Greta Thunberg called the deal 'a step forward, "
            "but not nearly enough.' The United States pledged $50 billion in climate finance "
            "for developing nations, which activists described as inadequate given decades of "
            "fossil fuel exploitation. European Union negotiators pushed for stronger language "
            "on coal phase-out, but faced resistance from fossil-fuel-dependent economies. "
            "Climate justice advocates emphasized that wealthy nations bear historical "
            "responsibility for the climate crisis and must do more. Indigenous communities "
            "from the Amazon and Pacific Islands submitted a joint declaration demanding "
            "immediate action on loss and damage financing. Binding enforcement mechanisms "
            "remain weak and voluntary commitments have historically been missed."
        ),
        "source_name": "The Guardian",
        "source_lean": "left",
        "published_at": "2025-11-12T09:00:00",
    },
    {
        "url": "https://apnews.com/test/climate-summit-center",
        "url_hash": "test_climate_center_001",
        "title": "Climate summit ends with 45% emissions reduction pledge by 2035",
        "body": (
            "Nations at the Global Climate Summit reached a binding agreement Saturday "
            "committing major economies to cut carbon emissions by 45% from 2005 levels "
            "by 2035. The accord, signed by 130 countries, establishes a $50 billion "
            "climate finance mechanism and creates a new loss and damage fund for "
            "climate-vulnerable nations. UN Secretary-General António Guterres called "
            "the deal 'a significant step' while acknowledging work remains. The agreement "
            "includes provisions for annual progress reviews and allows developing nations "
            "a five-year grace period on binding targets. Energy transition timelines vary "
            "by country income level. Negotiations ran 36 hours past the scheduled deadline "
            "as delegates resolved disputes over financing mechanisms and emission baseline years. "
            "China and India signed with reservations noted on the timeline for coal phase-down. "
            "The United States delegation confirmed all provisions. Implementation monitoring "
            "will be conducted by the UN Framework Convention on Climate Change secretariat."
        ),
        "source_name": "AP",
        "source_lean": "center",
        "published_at": "2025-11-12T10:15:00",
    },
    {
        "url": "https://foxnews.com/test/climate-summit-right",
        "url_hash": "test_climate_right_001",
        "title": "Climate summit deal will cost US economy $2 trillion while China gets a pass",
        "body": (
            "The Global Climate Summit agreement signed in Dubai this weekend is drawing "
            "sharp criticism from economists and energy industry analysts who warn the deal "
            "could cost the United States economy up to $2 trillion in lost productivity "
            "and increased energy prices by 2035. The accord commits developed nations to "
            "45% emissions cuts while giving China and India extended timelines, a disparity "
            "critics call fundamentally unfair. 'American workers and families will bear the "
            "burden while our competitors get a free pass,' said Senator Jim Barrasso. "
            "Oil and gas industry groups warned of job losses in energy-producing states. "
            "The $50 billion climate finance pledge was described as a wealth transfer "
            "from American taxpayers. Energy security advocates questioned whether renewable "
            "sources can reliably replace fossil fuel capacity within the aggressive timeline. "
            "Former President Donald Trump vowed to withdraw the United States from the "
            "agreement if elected. Carbon targets were seen as deeply unequal between "
            "developed and developing nations, threatening American industrial competitiveness."
        ),
        "source_name": "Fox News",
        "source_lean": "right",
        "published_at": "2025-11-12T11:30:00",
    },

    # ── Cluster 2: AI Regulation Bill ─────────────────────────────────────────
    {
        "url": "https://npr.org/test/ai-regulation-left",
        "url_hash": "test_ai_left_001",
        "title": "AI regulation bill raises civil liberties alarms as tech workers revolt",
        "body": (
            "A sweeping AI regulation bill advancing through Congress is sparking deep "
            "concerns among civil liberties organizations and tech workers who warn it "
            "fails to adequately protect marginalized communities from algorithmic bias. "
            "The American Civil Liberties Union said provisions allowing law enforcement "
            "to use AI surveillance tools without warrants are 'constitutionally dangerous.' "
            "Hundreds of Google, Amazon, and Microsoft employees signed an open letter "
            "calling on their employers to reject the bill's exemptions for corporate AI "
            "systems. Researchers found that facial recognition systems in the bill's "
            "approved list have error rates up to 34% higher for darker-skinned individuals. "
            "Senator Elizabeth Warren called for stronger algorithmic accountability "
            "provisions and civil rights protections. Labor unions representing data workers "
            "urged Congress to include employment protections as AI automation threatens "
            "millions of jobs. The Electronic Frontier Foundation described several bill "
            "provisions as 'a surveillance expansion dressed up as safety measures.'"
        ),
        "source_name": "NPR",
        "source_lean": "left",
        "published_at": "2025-11-10T14:00:00",
    },
    {
        "url": "https://reuters.com/test/ai-regulation-center",
        "url_hash": "test_ai_center_001",
        "title": "Senate AI regulation bill advances with bipartisan support, tech industry split",
        "body": (
            "The Artificial Intelligence Safety and Accountability Act passed the Senate "
            "Commerce Committee Tuesday on a 14-8 bipartisan vote, advancing to the full "
            "Senate for debate. The legislation would require risk assessments for high-risk "
            "AI systems, establish a new AI Safety Board within the FCC, and mandate "
            "transparency reports from major AI developers. OpenAI and Anthropic issued "
            "statements supporting the framework while Google and Meta expressed concerns "
            "about implementation timelines and compliance costs. The bill includes provisions "
            "for facial recognition oversight, autonomous vehicle standards, and healthcare "
            "AI audits. Civil liberties groups raised concerns about law enforcement exemptions. "
            "Industry groups welcomed liability protections but sought revisions to audit "
            "requirements. Congressional Budget Office estimates implementation costs at "
            "$4.2 billion over ten years. The bipartisan vote reflects growing congressional "
            "consensus on the need for federal AI oversight and governance."
        ),
        "source_name": "Reuters",
        "source_lean": "center",
        "published_at": "2025-11-10T15:30:00",
    },
    {
        "url": "https://dailywire.com/test/ai-regulation-right",
        "url_hash": "test_ai_right_001",
        "title": "New AI regulation bill hands government unprecedented control over tech sector",
        "body": (
            "A Senate bill to regulate artificial intelligence would give federal bureaucrats "
            "sweeping power to dictate how American companies develop and deploy AI technology, "
            "critics warn. The Artificial Intelligence Safety and Accountability Act establishes "
            "a new federal AI Safety Board with authority to approve or block AI products, "
            "a power conservatives describe as government overreach that could cripple American "
            "innovation. 'This is Silicon Valley's worst nightmare dressed up as safety policy,' "
            "said tech analyst Marc Andreessen. The legislation imposes compliance costs "
            "estimated at $800,000 annually per company, costs that will fall disproportionately "
            "on startups while big tech corporations like Meta and Google absorb them easily. "
            "China, facing no such regulatory burden, could overtake American AI leadership "
            "within a decade under these constraints. Heritage Foundation analysts warned the "
            "bill represents 'the most aggressive federal intrusion into the technology sector "
            "in American history' and would hand big tech incumbents a regulatory moat."
        ),
        "source_name": "Daily Wire",
        "source_lean": "right",
        "published_at": "2025-11-10T16:45:00",
    },

    # ── Cluster 3: Federal Reserve Interest Rate Decision ──────────────────────
    {
        "url": "https://huffpost.com/test/fed-rates-left",
        "url_hash": "test_fed_left_001",
        "title": "Fed rate hold punishes working families as housing crisis deepens",
        "body": (
            "The Federal Reserve's decision to hold interest rates steady at 5.25% is "
            "drawing criticism from housing advocates and economists who say the policy "
            "continues to crush working-class Americans' dreams of homeownership. "
            "With 30-year mortgage rates still hovering near 7.5%, first-time buyers "
            "are effectively locked out of the housing market, while wealthy investors "
            "with existing real estate holdings benefit from rising property values. "
            "Rents in major cities have increased 18% over the past two years, forcing "
            "many low-income families to spend over 50% of their income on housing. "
            "Progressive economists argued the Fed's inflation obsession is protecting "
            "corporate profit margins at the expense of workers. Senator Bernie Sanders "
            "called on Fed Chair Jerome Powell to prioritize maximum employment over "
            "inflation targets that primarily serve Wall Street interests. Unemployment "
            "has risen 0.4 percentage points since the rate hikes began, with Black and "
            "Hispanic workers disproportionately affected by the monetary tightening cycle."
        ),
        "source_name": "HuffPost",
        "source_lean": "left",
        "published_at": "2025-11-13T18:00:00",
    },
    {
        "url": "https://bbc.com/test/fed-rates-center",
        "url_hash": "test_fed_center_001",
        "title": "Federal Reserve holds rates at 5.25% as inflation shows mixed signals",
        "body": (
            "The Federal Reserve left its benchmark interest rate unchanged at 5.25% Wednesday "
            "as policymakers weighed moderating inflation against signs of economic softening. "
            "Fed Chair Jerome Powell said the committee remains 'data dependent' and is "
            "monitoring labor market conditions closely. Core PCE inflation fell to 2.7% "
            "annually, above the Fed's 2% target but down from 3.4% a year ago. "
            "The unemployment rate rose to 4.2%, the highest since 2021. Markets had priced "
            "in a 30% probability of a rate cut, rising to 65% following the announcement. "
            "Powell said a December cut is 'on the table' if inflation continues declining. "
            "GDP growth slowed to 1.8% in the third quarter. Housing starts fell 8% in "
            "October. Consumer spending remained resilient despite high borrowing costs. "
            "Three dissenting Federal Reserve members favored a 25 basis point cut, "
            "citing weakening employment indicators and slowing economic momentum."
        ),
        "source_name": "BBC",
        "source_lean": "center",
        "published_at": "2025-11-13T19:15:00",
    },
    {
        "url": "https://washingtonexaminer.com/test/fed-rates-right",
        "url_hash": "test_fed_right_001",
        "title": "Fed forced to hold rates as Biden's inflation legacy haunts the economy",
        "body": (
            "The Federal Reserve was compelled to maintain its restrictive 5.25% interest "
            "rate Wednesday, a direct consequence of the inflationary spending spree that "
            "defined the Biden administration's economic legacy. Core inflation at 2.7% "
            "remains 35% above the Fed's target, the result of trillions in deficit spending "
            "and the so-called Inflation Reduction Act, which economists widely predicted "
            "would accelerate inflation. 'This is the bill coming due for four years of "
            "reckless fiscal policy,' said economist Stephen Moore. The Federal Reserve's "
            "independence has been questioned as it navigated political pressure ahead of "
            "the 2024 election. American businesses face the highest borrowing costs in "
            "16 years while competing against foreign firms in lower-rate environments. "
            "Small business loan applications fell 22% this quarter. Republicans on the "
            "Joint Economic Committee argued regulatory rollbacks would be more effective "
            "inflation remedies than blunt monetary policy tools from Jerome Powell."
        ),
        "source_name": "Washington Examiner",
        "source_lean": "right",
        "published_at": "2025-11-13T20:00:00",
    },
]

# Pre-defined entities for each article (keyed by url_hash).
# Avoids running spaCy during seeding — entities still support the entity-rescue
# clustering tier for borderline cosine matches.
SEED_ENTITIES: dict[str, list[dict]] = {
    "test_climate_left_001": [
        {"text": "United States", "normalized": "united states", "label": "GPE"},
        {"text": "European Union", "normalized": "european union", "label": "ORG"},
        {"text": "Greta Thunberg", "normalized": "greta thunberg", "label": "PERSON"},
        {"text": "Dubai", "normalized": "dubai", "label": "GPE"},
    ],
    "test_climate_center_001": [
        {"text": "United States", "normalized": "united states", "label": "GPE"},
        {"text": "China", "normalized": "china", "label": "GPE"},
        {"text": "India", "normalized": "india", "label": "GPE"},
        {"text": "António Guterres", "normalized": "antónio guterres", "label": "PERSON"},
        {"text": "Dubai", "normalized": "dubai", "label": "GPE"},
    ],
    "test_climate_right_001": [
        {"text": "United States", "normalized": "united states", "label": "GPE"},
        {"text": "China", "normalized": "china", "label": "GPE"},
        {"text": "India", "normalized": "india", "label": "GPE"},
        {"text": "Jim Barrasso", "normalized": "jim barrasso", "label": "PERSON"},
        {"text": "Donald Trump", "normalized": "donald trump", "label": "PERSON"},
        {"text": "Dubai", "normalized": "dubai", "label": "GPE"},
    ],
    "test_ai_left_001": [
        {"text": "Google", "normalized": "google", "label": "ORG"},
        {"text": "Amazon", "normalized": "amazon", "label": "ORG"},
        {"text": "Microsoft", "normalized": "microsoft", "label": "ORG"},
        {"text": "Elizabeth Warren", "normalized": "elizabeth warren", "label": "PERSON"},
        {"text": "American Civil Liberties Union", "normalized": "american civil liberties union", "label": "ORG"},
    ],
    "test_ai_center_001": [
        {"text": "OpenAI", "normalized": "openai", "label": "ORG"},
        {"text": "Anthropic", "normalized": "anthropic", "label": "ORG"},
        {"text": "Google", "normalized": "google", "label": "ORG"},
        {"text": "Meta", "normalized": "meta", "label": "ORG"},
        {"text": "Senate", "normalized": "senate", "label": "ORG"},
    ],
    "test_ai_right_001": [
        {"text": "Meta", "normalized": "meta", "label": "ORG"},
        {"text": "Google", "normalized": "google", "label": "ORG"},
        {"text": "Heritage Foundation", "normalized": "heritage foundation", "label": "ORG"},
        {"text": "Marc Andreessen", "normalized": "marc andreessen", "label": "PERSON"},
        {"text": "China", "normalized": "china", "label": "GPE"},
        {"text": "Senate", "normalized": "senate", "label": "ORG"},
    ],
    "test_fed_left_001": [
        {"text": "Federal Reserve", "normalized": "federal reserve", "label": "ORG"},
        {"text": "Jerome Powell", "normalized": "jerome powell", "label": "PERSON"},
        {"text": "Bernie Sanders", "normalized": "bernie sanders", "label": "PERSON"},
        {"text": "Wall Street", "normalized": "wall street", "label": "ORG"},
    ],
    "test_fed_center_001": [
        {"text": "Federal Reserve", "normalized": "federal reserve", "label": "ORG"},
        {"text": "Jerome Powell", "normalized": "jerome powell", "label": "PERSON"},
        {"text": "United States", "normalized": "united states", "label": "GPE"},
    ],
    "test_fed_right_001": [
        {"text": "Federal Reserve", "normalized": "federal reserve", "label": "ORG"},
        {"text": "Jerome Powell", "normalized": "jerome powell", "label": "PERSON"},
        {"text": "Biden", "normalized": "biden", "label": "PERSON"},
        {"text": "Stephen Moore", "normalized": "stephen moore", "label": "PERSON"},
        {"text": "United States", "normalized": "united states", "label": "GPE"},
    ],
}
