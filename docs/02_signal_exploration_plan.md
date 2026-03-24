# Signal Exploration Plan

## Purpose

This document lays out the first serious plan for transcript-level feature development. The goal is to guide early feature engineering without pretending that the right signal definitions are already known.

## Why Sentiment Is Not the Same as Communication Style

Sentiment asks whether language is favorable, unfavorable, or neutral. Communication style asks how management communicates.

Those are related but different constructs. For example:

- a manager can sound positive but still vague
- a manager can sound cautious without being evasive
- a manager can provide negative information in a direct and transparent way
- a manager can use neutral language while deflecting responsibility

For this project, sentiment should therefore be treated as a benchmark family of features, not as a substitute for communication-style measures.

## Candidate Signal Families

### 1. Sentiment

Purpose:

- provide a benchmark for comparison
- avoid attributing standard tone effects to style effects

Examples:

- positive, negative, neutral sentiment
- net tone only if the component scores are also preserved

### 2. Directness or Specificity

Purpose:

- capture whether management speaks concretely and explicitly rather than vaguely

Possible indicators:

- numerical precision
- specific operational references
- explicit action statements
- concrete explanations rather than generic framing

### 3. Evasiveness or Deflection

Purpose:

- capture whether management avoids answering directly or shifts away from the substance of a question

Possible indicators:

- non-answer phrases
- redirecting language
- broad reframing after a pointed question
- repeated use of generic language in analyst exchange

### 4. Hedging or Uncertainty

Purpose:

- capture qualified, cautious, or probabilistic language

Possible indicators:

- modal verbs
- soft commitments
- uncertainty markers
- forecast caveats beyond standard safe-harbor language

### 5. Vagueness

Purpose:

- capture imprecision in descriptions, timing, responsibility, or outcomes

Possible indicators:

- indefinite quantifiers
- non-specific time references
- ambiguous descriptors
- weakly bounded operational statements

### 6. Transparency

Purpose:

- capture whether management explains assumptions, mechanisms, and details clearly

Possible indicators:

- explicit disclosure language
- detailed breakdowns
- discussion of drivers and assumptions
- acknowledgment of uncertainty with substance rather than deflection

### 7. Internal vs External Attribution

Purpose:

- capture whether outcomes are attributed to management choices, internal execution, market conditions, macro shocks, or other external forces

Possible indicators:

- first-person responsibility language
- passive constructions
- mentions of market, customer, regulatory, or macro explanations
- contrasting internal and external causal framing

## Measurement Approaches

### Dictionary or Lexicon Methods

Useful for:

- initial prototypes
- transparent auditing
- interpretable feature definitions

Examples:

- hedging lexicons
- vagueness terms
- transparency and disclosure terms
- attribution terms for internal and external causes

Main strengths:

- easy to inspect
- easy to revise
- low setup cost

Main weaknesses:

- context-insensitive
- can overcount boilerplate
- may confuse legal language with true communication style

### N-Grams and Phrase Patterns

Useful for:

- direct question-answer phrases
- deflection patterns
- recurring constructions that are not well captured by single words

Examples:

- "as you know"
- "hard to say"
- "we are not going to"
- "what I would say is"
- "let me take that at a high level"

Main strengths:

- better than single-word counts for some styles
- useful in Q&A analysis

Main weaknesses:

- brittle to wording variation
- easy to overfit to a small sample of phrases

### Sentence-Level Classification

Useful for:

- tagging sentences as hedged, direct, evasive, or explanatory
- aggregating sentence-level labels to transcript-level features

Possible routes:

- weak supervision
- hand-labeled pilot sample
- zero-shot classification for early experiments

Main strengths:

- better context sensitivity
- more natural fit for complex styles

Main weaknesses:

- requires label design and validation
- higher setup cost
- easy to create noisy labels

### Embeddings and Semantic Similarity

Useful for:

- clustering similar responses
- comparing management answers to prototypical direct or evasive statements
- grouping thematic content before style scoring

Main strengths:

- flexible and expressive
- can capture semantic similarity beyond exact wording

Main weaknesses:

- less interpretable
- requires stronger validation before being used in finance tests

## Recommended Segmentation: Prepared Remarks vs Q&A

Where possible, the first serious feature pass should preserve separate views for:

- full transcript
- prepared remarks
- Q&A
- management responses within Q&A

Why this matters:

- prepared remarks are partially scripted
- Q&A is often where evasiveness, deflection, and accountability become more visible
- some style signals may be much stronger in interactive segments than in prepared remarks

If section separation is not immediately available, the limitation should be documented explicitly.

## Measurement Risks and Pitfalls

- safe-harbor and legal boilerplate can inflate uncertainty-style measures
- transcript length can dominate raw counts
- topic mix can masquerade as communication style
- sentiment and style may overlap unless explicitly separated
- dictionary signals can be noisy across industries and call formats
- deflection is difficult to detect without question context
- aggregation can hide meaningful within-transcript heterogeneity

## Proposed First-Pass Prototype Plan

### Phase 1: Interpretable Baselines

- define seed dictionaries for hedging, vagueness, directness, transparency, attribution, and evasiveness proxies
- normalize counts by transcript or section length
- inspect top-scoring transcripts manually for face validity
- keep sentiment outputs separate

### Phase 2: Phrase-Level Enrichment

- add phrase-pattern features for common Q&A deflection or qualification language
- compare single-word and phrase-based measures

### Phase 3: Conditional Model-Assisted Work

- consider sentence-level classification or embedding-based methods only after the baseline dictionaries are documented and inspected
- use model-assisted methods to refine or challenge baseline measures rather than replace interpretability immediately

## Short-Run Deliverable

The expected near-term output is a documented, auditable set of candidate transcript features suitable for EDA and later baseline finance tests. It is not proof that any particular signal predicts returns.
