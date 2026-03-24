# Research Question and Empirical Design Memo

## Purpose

This memo frames the likely research question and the first-pass empirical design. It is a design document for planning the data pipeline and signal construction, not a claims document.

## Candidate Research Question

Do earnings call transcripts contain communication-style information that is distinct from sentiment and relevant for asset-pricing outcomes?

## Narrowed Working Question

After controlling for sentiment and standard firm-level information, do transcript-based communication-style measures help explain short-window market reactions around earnings calls or subsequent return patterns?

## Economic Intuition

Earnings calls do more than convey numeric information. They also reveal how management communicates under scrutiny. Communication style may matter because:

- investors may update beliefs not only from what is said, but from how it is said
- hedging, directness, and attribution may affect perceived credibility
- evasiveness in Q&A may signal information frictions or weak disclosure quality
- transparency may affect how quickly the market incorporates information

If these style dimensions contain incremental information, they may help explain differences in immediate reaction, post-event drift, or related outcomes.

## Why This Is Interesting in Finance

- earnings calls are central information events
- transcript data are rich, scalable, and increasingly used in empirical finance
- much existing work emphasizes tone or sentiment, leaving room to study communication style more explicitly
- a design that separates sentiment from style is more conceptually disciplined than a one-score text summary

## Key Hypotheses

These are candidate hypotheses for later testing, not established findings.

### H1

Communication-style measures contain information beyond benchmark sentiment measures.

### H2

The strongest style effects, if they exist, are more likely to appear in Q&A than in prepared remarks.

### H3

More direct and transparent communication may be associated with different immediate market reactions than more hedged or evasive communication.

### H4

Internal versus external attribution patterns may help explain how investors interpret management credibility or responsibility.

## Proposed Dependent Variables

Potential outcome variables for later analysis:

- abnormal return around the earnings call
- short-window cumulative abnormal return
- post-event drift over a chosen horizon
- future return measures over short horizons
- possibly later extensions to fundamentals or analyst-related outcomes

The initial emphasis should remain on returns-based outcomes because they align most directly with the working finance question.

## Proposed Independent Variables

Primary text variables:

- sentiment measures
- hedging or uncertainty measures
- vagueness measures
- directness or specificity measures
- transparency measures
- evasiveness or deflection measures
- internal versus external attribution measures

Possible representation levels:

- full-transcript measures
- prepared-remarks measures
- Q&A measures
- management-response-only measures

## Event Study Design Sketch

### Goal

Test whether transcript-based features are associated with short-window market reactions around the earnings call.

### Baseline design

- unit of analysis: firm-call event
- event date: to be finalized after data validation
- event windows: start with narrow windows such as `[-1, +1]` or `[0, +1]`
- benchmark: market-adjusted or model-based abnormal returns, to be specified later

### Core specification idea

Compare:

1. a baseline model with sentiment and standard controls
2. an expanded model that adds communication-style variables

The main question is whether style features add explanatory power beyond sentiment.

## Panel or Cross-Sectional Design Sketch

### Goal

Test whether transcript-based features help explain future return variation after the call.

### Baseline design

- unit of analysis: firm-quarter or firm-call
- dependent variable: future return over a defined post-event horizon
- core independent variables: sentiment plus style features
- estimation style: panel or repeated cross-sections, depending on sample structure and control needs

### Main comparison logic

- sentiment-only specification
- sentiment-plus-style specification
- sensitivity to alternative feature definitions and sample restrictions

## Likely Controls

The first baseline specifications will likely need some combination of:

- earnings surprise
- firm size
- momentum or prior returns
- return volatility
- industry controls
- time controls
- transcript length or word count

Possible later additions:

- analyst coverage
- liquidity
- fixed effects where sample structure supports them

## Likely Robustness Checks

- separate prepared remarks from Q&A
- exclude or downweight boilerplate-heavy transcripts
- use alternative event windows
- compare dictionary-based and model-assisted style measures
- winsorize extreme feature values
- test whether results survive length normalization choices
- inspect whether style variables are overly collinear with sentiment

## Limitations and Identification Concerns

- communication style may proxy for underlying firm condition rather than adding independent information
- management style may vary systematically by industry, firm maturity, or business model
- transcript sections may mix scripted and unscripted language
- some style proxies may mainly capture legal or disclosure conventions
- event-date misalignment can contaminate return tests
- signal construction choices may be noisy, especially early in the project

## What Would Make the Idea More Paper-Worthy

- a clearly motivated construct that is distinct from standard tone measures
- convincing separation of communication style from sentiment
- careful treatment of Q&A versus prepared remarks
- robust evidence that style adds incremental information in economically meaningful settings
- strong validation that the constructed signals are interpretable and not dominated by boilerplate or length effects

## Immediate Design Decisions Still Needed

- final event date definition
- canonical observation unit
- first benchmark sentiment method
- first-pass style feature definitions
- returns data source and abnormal-return construction
