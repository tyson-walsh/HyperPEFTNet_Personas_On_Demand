#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hypernetwork_feature_builder_10000.py
======================================

Purpose
-------
Build a **global-static** (author-level) feature representation used to condition a
small hypernetwork that generates additive PEFT weight offsets δθ for a frozen
backbone (e.g., HyperPEFT-LoRA). The builder produces:

1) an **author table** (one row per target_user_id) containing global-static
   features (gstat_*), and

2) one or more **per-thread global tables** (one row per thread gid) produced by
   replicating the author vector onto each thread row for that user.

The resulting per-thread global table is the conditioning input: when training or
running inference, each thread gid is paired with the author’s fixed feature vector
g (global-static), and the hypernetwork maps g → δθ.

Global-static here means: the feature depends only on the target_user_id’s corpus
within the dataset window (and the configured aggregation split), and it does not
vary from thread to thread for that author.

Dataset semantics (critical)
----------------------------
The dataset is constructed from conversation “chains” and uses a thread anchor id:

• target_user_id is the **anchor user for the thread** (the target author whose
  reply is predicted). It is stamped onto every row in the chain, including
  context rows written by other people.

Therefore:
• Rows where group_label ends with "_target" are the only rows that are guaranteed
  to be written by the target_user_id. These rows represent the author’s own text.
• Context rows must not be treated as authored by target_user_id when computing
  stylistic/personality/circadian “author fingerprint” statistics.

This builder follows that rule:
• Author-fingerprint features are computed using ONLY "_target" rows.
• Reply-delay features require both context + target timestamps and are computed
  by grouping rows by gid and comparing the target timestamp to the latest context
  timestamp, then assigning the delay to the target_user_id of the target row.

Leakage-safe evaluation mode
----------------------------
The script supports computing author vectors from a chosen split set:

• --agg_split train    (default): author vectors use train data only.
• --agg_split trainval : author vectors use train+val data.
• --agg_split all      : author vectors use train+val+test data.

Regardless of --agg_split, per-thread tables can be emitted for train/val/test gids.
When using --agg_split train, the per-thread tables for val/test are conditioned on
author vectors computed without using val/test targets. If a target_user_id appears
in val/test but not in train, the script fills its features with zeros (which
correspond to “mean” for z-scored scalars), preserving schema and avoiding hard
failures.

Inputs and expected columns
---------------------------
The script reads three Parquet splits (paths configurable via CLI):

• train_data_10000.parquet
• val_data_10000.parquet
• test_data_10000.parquet

Required columns:
• gid            : int. Thread identifier. Must be positive (> 0). Invalid rows are dropped.
• target_user_id : int. Numeric id of the thread’s target author. Must be positive (> 0). Invalid rows are dropped.
• text           : str. Normalized comment text.
• created_utc    : Timestamp-like. Coerced to timezone-aware UTC. NaT allowed.
• group_label    : str. Rows whose label ends with "_target" are the target author’s
                   reply for that gid.

Strictness note (current behavior):
• group_label must be present and each relevant split must contain at least one "_target" row;
  otherwise the script raises an error.
• In Stage B replication outputs, the script requires **exactly one** "_target" row per gid
  in the emitted split(s). If multiple "_target" rows exist for a gid, the replication stage
  raises an error (instead of silently choosing one).

Optional columns (used when present):
• subreddit column named one of:
  subreddit, subreddit_name, subreddit_name_prefixed, subreddit_id, sr, sr_name
• score : numeric. Engagement proxy; used only for simple per-author stats.

Outputs and side-cars
---------------------
Author table (always):
• author_static_10000.parquet
  One row per target_user_id containing gstat_* columns (and target_user_id).

Per-thread global table(s) (always produces the combined table):
• global_features_10000.parquet
  One row per gid across train+val+test (one target row per gid), containing:
  gid, target_user_id, and the replicated gstat_* columns.

Optional split-specific global tables (enabled by --split_outputs valtest or all):
• global_features_val_10000.parquet   (val gids only)
• global_features_test_10000.parquet  (test gids only)
• global_features_train_10000.parquet (train gids only; only when --split_outputs all)

Side-cars:
• feature_norm_stats_10000.json
  Mean/std for each scalar gstat_* column computed over the author table (authors
  equally weighted). These stats are used to z-score scalar features at build time.
• *_cols.json
  Column order side-car for each emitted global parquet (schema order as written).

Sharding, resume, merge
-----------------------
Stage A (author feature computation) runs unsharded to guarantee exactly one
author vector per user.

Stage B (replication/writing) can be sharded via --shards=N:
• Each worker keeps rows where (gid % N == rank).
• Each worker writes a part file with suffix _w<rank>.
• A checkpoint file records last processed gid per output per rank:
  feature_builder_ckpt_<output_stem>_<rank>.json
• After workers finish, the script merges all parts into the canonical output name.
  (Current implementation merges shards in a streaming fashion to reduce RAM spikes.)

Computation overview
--------------------
Stage A: build author_static_10000.parquet
1) Load the aggregation corpus defined by --agg_split.
2) Extract target rows (group_label endswith "_target") into df_tgt.
3) Compute:
   • sentiment polarity per post + per-user mean/variance and global baseline gap
   • lexical/stylistic ratios per user (token- and char-normalized)
   • subreddit concentration metrics per user
   • circadian activity histogram over UTC hours using target timestamps
   • reply-delay distribution using gid-level target-vs-context timestamps
   • personality (OCEAN) vectors from a classifier (multiple output views)
   • optional BEAST-GB meta-features from gradient boosting on numeric globals
4) Compute derived personality centered traits and z-scored variants across authors
   for multiple personality output views (primary/sigmoid/softmax/logits).
5) Z-score all scalar global features across authors and write normalization stats.

Stage B: build per-thread global tables
1) Load (gid, target_user_id, group_label) from requested split(s).
2) Require exactly one target row per gid (error if duplicates).
3) Replicate the author’s gstat_* vector onto that gid’s row.
4) Write parquet in large chunks, merge shards, and write *_cols.json.

Tokenization and normalization conventions
------------------------------------------
Two tokenization views are used intentionally:

A) “Word tokens” (lexicon-based ratios):
   toklist(text) := Unicode-aware regex of letter tokens with internal apostrophes,
                    casefolded (Unicode-lowercased) and with curly apostrophes normalized.
   • strips numbers/punctuation/emoji/underscores
   • keeps apostrophes in contractions as part of tokens (both ' and ’)
   Used for profanity/abuse/stance lexicons, stopwords, and bigram redundancy.

B) “Whitespace tokens” (length and TTR):
   text.split() on whitespace.
   Used for:
   • gstat_user_len_mean (mean length per post)
   • gstat_user_ttr (unique whitespace tokens / total whitespace tokens across the author corpus)

Character normalization:
• Per-character ratios use raw string length len(text) (including punctuation/emoji).

Time conventions:
• created_utc is coerced to UTC datetime; hour bins and weekend checks are in UTC.

Rounding and determinism:
• List-valued vectors are rounded to ROUND_DP (default 6) for stable serialization.
• Scalar columns are stored as float32 after z-scoring.
• Fixed random seed is used for the global sentiment baseline sample.

Sentiment caching (current behavior):
• Polarity scoring truncates inputs to the first 256 whitespace tokens.
• The polarity cache key uses the *truncated* text actually scored (not the full original text).
• Cache capacity is controlled by env var POLAR_CACHE_MAX (default 500000). Set to 0 to disable caching.

GLOBAL-STATIC FEATURES (author fingerprint gstat_*)
---------------------------------------------------

A. Personality block (OCEAN; list-valued, not scalar-z-scored)
Why: OCEAN-style traits capture broad stable tendencies in language production
(e.g., extraversion/agreeableness cues). Logits retain calibration signal.

For each author:
• Select up to 200 target posts deterministically (sorted by created_utc then gid).
• Run a sequence classification model:
  - logits := model(**tokenized).logits
  - always compute:
      sigmoid_raw := sigmoid(logits_sel)  (independent per-trait probabilities)
      softmax_raw := softmax(logits_sel)  (distribution over the 5 selected traits; sums to 1)
      mean_logits := mean(logits_sel)     (regression-friendly summary)
• Remap label indices to canonical order [A, O, C, E, N] using model config id2label
  heuristics (best-effort mapping) OR via explicit override --persona_idx_map.

Primary personality vector selection (current behavior):
• gstat_personality_raw is a “primary” vector chosen by --persona_activation:
  - sigmoid     → primary = sigmoid_raw
  - softmax     → primary = softmax_raw
  - regression  → primary = mean_logits
  - auto        → inferred from model config when available (may choose regression)

Outputs (all list[5]):
1) gstat_personality_raw
   Primary author-level personality vector (see selection rules above).

2) gstat_personality_sigmoid_raw
   Mean sigmoid(logits_sel) over sampled posts, in A/O/C/E/N order.

3) gstat_personality_softmax_raw
   Mean softmax(logits_sel) over sampled posts, in A/O/C/E/N order.

4) gstat_personality_logits
   Mean selected logits over sampled posts, in A/O/C/E/N order.

Derived centered + z variants (list[5], not part of scalar z-scoring pass):
• For each of the four base vectors above, compute:
  - *_traits: median-centered across authors (per-coordinate)
  - *_z     : z-scored across authors (per-coordinate, std + 1e-6)

These are emitted as:
• gstat_personality_traits
• gstat_personality_z
• gstat_personality_sigmoid_traits
• gstat_personality_sigmoid_z
• gstat_personality_softmax_traits
• gstat_personality_softmax_z
• gstat_personality_logits_traits
• gstat_personality_logits_z

B. Sentiment and baseline gap (scalar; z-scored)
Why: coarse affective tone and volatility are stable author cues and interact with
other style features.

Sentiment polarity per post:
• Use a sentiment model returning POS/NEG scores.
• Polarity(text) := P(POS) - P(NEG).
• Inputs are truncated to 256 whitespace tokens before scoring.
• Results are memoized in an LRU cache keyed by the truncated (scored) text string
  (cache size controlled by POLAR_CACHE_MAX; 0 disables caching).

Global baseline:
• Sample up to 1000 target posts from df_tgt with random_state=142.
• global_sent := mean polarity over that sample.

5)  gstat_gap_sentiment    = mean_user_polarity - global_sent
6)  gstat_user_sent_mean   = mean polarity over the author’s target posts
7)  gstat_user_sent_var    = variance (ddof=0) of polarity over the author’s target posts

C. Length, lexical diversity, posting rate (scalar; z-scored)
Why: verbosity and lexical diversity are stable style signals; post-rate proxies
activity intensity.

8)  gstat_user_len_mean
    Mean whitespace token count per target post:
    mean_i len(text_i.split()).

9)  gstat_user_ttr
    Type–token ratio over whitespace tokens across the author corpus:
    unique_whitespace_tokens / total_whitespace_tokens.

10) gstat_user_post_rate
    Posts per day across the data-derived corpus window:
    (# target posts for author) / corpus_days.
    corpus_days is computed from actual data timestamps (max - min + 1);
    falls back to CORPUS_DAYS constant (2010-01-01..2016-01-01) if insufficient data.

D. Subreddit concentration (scalar; z-scored)
Why: topical breadth and community concentration are strong behavioral priors.

Subreddit column selection:
• The first available column among:
  subreddit, subreddit_name, subreddit_name_prefixed, subreddit_id, sr, sr_name.
If absent, all values default to 0.

For each author, let p[s] be normalized subreddit frequencies over target posts.

11) gstat_user_subreddit_entropy
    H(p) = -∑_s p[s] log2(p[s] + 1e-12)

12) gstat_user_sr_herfindahl
    HHI(p) = ∑_s p[s]^2

13) gstat_user_sr_max_share
    max_s p[s]

E. Stylistic and pragmatic ratios (scalar; z-scored)
Why: compact, interpretable cues capture punctuation use, stance, politeness,
register, and rhetorical habits.

Unless otherwise stated, “per-post averages” mean the ratio is computed per post
and then averaged across posts (each post equal weight). “Corpus ratios” mean
the numerator and denominator are computed across all target posts jointly
(length-weighted).

14) gstat_punct_ratio  (per-post average; per-char)
    Fraction of characters that are not alphanumeric/whitespace, excluding emoji
    (to reduce double-counting between punctuation and emoji features).

15) gstat_question_ratio (per-post average; boolean)
    Fraction of target posts whose stripped text ends with '?'.

16) gstat_caps_ratio (per-post average; per-token)
    Fraction of whitespace tokens that are ALL CAPS and length > 1.

17) gstat_profanity_ratio (per-post average; per-token)
    Profanity lexicon match rate using toklist tokens:
    matches / max(1, #toklist_tokens) per post, then averaged.

18) gstat_abuse_ratio (corpus ratio; per-token)
    Abuse/insult lexicon match rate over all toklist tokens across the author corpus:
    total_matches / max(1, total_toklist_tokens).

19) gstat_firstperson_ratio (per-post average; per-token)
    Rate of {i, me, my, mine} per toklist token, averaged over posts.

20) gstat_secondperson_ratio (per-post average; per-token)
    Rate of {you, your, yours, yourself, ... incl. u/ya/yall} per toklist token,
    averaged over posts.

21) gstat_readability_fk (median over posts)
    Median Flesch–Kincaid grade over target posts.
    Uses textstat if available, otherwise a syllable-based fallback:
      0.39*(words/sentences) + 11.8*(syllables/words) - 15.59

22) gstat_weekend_ratio (per-post average; boolean)
    Fraction of target posts with created_utc weekday >= 5 (Saturday/Sunday).
    Missing timestamps contribute 0.0.

23) gstat_link_ratio (per-post average; boolean)
    Fraction of posts containing "<URL>" OR a literal http:// / https:// OR "www." substring.

24) gstat_hedge_ratio (corpus ratio; per-token)
    (single-word hedge hits + multi-word hedge phrase occurrences) / total toklist tokens.
    Single-word hedges include e.g., maybe, perhaps, probably, kinda, sorta.
    Multi-word hedges include e.g., “i think”, “it seems”, “kind of”.

25) gstat_intensifier_ratio (corpus ratio; per-token)
    Intensifier tokens (e.g., very, really, extremely, literally, so, super) / total toklist tokens.

26) gstat_modal_ratio (corpus ratio; per-token)
    Modal verb tokens (would, could, should, might, may, can, must, ought) / total toklist tokens.

27) gstat_negation_ratio (corpus ratio; per-token)
    Negation tokens/markers (not, no, never, don't, can't, won't, ...) / total toklist tokens.

28) gstat_subjectivity_ratio (corpus ratio; per-token)
    Subjective stance cues (e.g., amazing, awful, love, hate, think, feel, believe) / total toklist tokens.

29) gstat_agreement_ratio (corpus ratio; per-token, with multi-word support)
    (single-word agreement hits + multi-word phrase hits) / total toklist tokens.
    Single-word: {yes, yeah, yep, correct, true, agree, exactly, indeed, definitely}.
    Multi-word: {"for sure", "no doubt", "spot on", "you're right", "i agree",
                 "same here", "fair point", "good point", "well said"}.

30) gstat_disagreement_ratio (corpus ratio; per-token, with multi-word support)
    (single-word disagreement hits + multi-word phrase hits) / total toklist tokens.
    Single-word: {nope, nah, wrong, disagree, bs, nonsense, incorrect, mistaken,
                  doubtful, debatable, questionable, unlikely, hardly}.
    Multi-word: {"not true", "i disagree", "don't agree", "you're wrong",
                 "not right", "absolutely not", "no way", "i doubt",
                 "not really", "that's not"}.

31) gstat_slang_ratio (corpus ratio; per-token)
    Netspeak tokens (lol, lmao, idk, imo, btw, tbh, ...) / total toklist tokens.

32) gstat_emoji_ratio (corpus ratio; per-token)
    Unicode emoji matches / total toklist tokens.
    Emoji detection uses a broad Unicode-range regex and counts codepoints in matched runs.

33) gstat_emoticon_ratio (corpus ratio; per-char)
    ASCII emoticon occurrences (":)", ":-(", "XD", ...) / total characters.
    (Denominator changed from toklist tokens to characters for dimensional
    consistency — emoticon_count returns char-level counts.)

34) gstat_contraction_ratio (corpus ratio; per-token)
    Contraction regex matches per toklist token:
      matches of pattern \\b\\w+'(m|re|d|ll|ve|s|t)\\b / total toklist tokens.

35) gstat_ellipsis_ratio (corpus ratio; per-char)
    Count of literal "..." occurrences / total characters across all posts.

36) gstat_punct_burst_ratio (corpus ratio; per-char)
    Count of repeated punctuation bursts (e.g., "!!", "???") / total characters.

37) gstat_sarcasm_ratio (per-post average; boolean)
    Fraction of posts containing at least one sarcasm marker (e.g., "/s",
    "yeah right", "as if", "sure..."). Converted from substring-count/token-count
    to boolean per-post rate for dimensional consistency.

38) gstat_quote_ratio (corpus ratio; per-char)
    (count of '"' + count of "'") across posts / total characters.

39) gstat_avg_word_len (corpus average)
    Mean length of toklist tokens (letters/apostrophes only).

40) gstat_long_word_ratio (corpus ratio; per-token)
    Fraction of toklist tokens with length >= 6.

41) gstat_stopword_ratio (corpus ratio; per-token)
    Fraction of toklist tokens in a compact, dependency-free stopword list.

42) gstat_rep_bigram_ratio (corpus redundancy)
    Bigram redundancy over toklist tokens:
      1 - ( #unique_bigrams / #total_bigrams )
    where bigrams are adjacent toklist token pairs across the concatenated corpus.
    Interprets higher values as more repetition or formulaic phrasing.

F. Circadian activity (list[24] + derived scalars)
Why: coarse temporal habits are stable behavioral cues.

Hour histogram source:
• ONLY target rows’ created_utc are used (UTC hour of day).

43) gstat_hour_hist (list[24])
    Let c[h] be the count of target posts in hour h. Output:
      p[h] = c[h] / sum_h c[h]
    If the author has no valid timestamps, all bins are 0.0.

44) gstat_hour_entropy (scalar)
    Entropy of p[h]:
      -∑_h p[h] log2(p[h] + 1e-12)

45) gstat_nocturnal_ratio (scalar)
    ∑_{h=0}^{5} p[h]  (midnight–5:59 UTC share)

46) gstat_peak_hour (int32; not z-scored)
    argmax_h p[h] in {0..23}. If all bins are zero, argmax is 0.

47) gstat_circadian_mean (scalar)
    Circular mean hour robust to wrap-around:
    • θ_h = 2πh/24
    • x = ∑ p[h] cos θ_h, y = ∑ p[h] sin θ_h
    • θ̄ = atan2(y, x) mapped to [0, 2π)
    • mean_hour = 24 θ̄ / (2π)

G. Reply-delay distribution (scalar; z-scored)
Why: responsiveness vs deliberation is a stable interaction style cue.

Computation source:
• Uses all rows in the aggregation corpus (context + target) grouped by gid.
• For each gid:
  - identify target row(s) (group_label endswith "_target");
    if multiple exist, the latest target timestamp is selected deterministically.
  - reply_time := created_utc of selected target row
  - ctx_time   := max created_utc among non-target rows
  - delay_min  := (reply_time - ctx_time) in minutes, kept if finite and >= 0
• Delays are assigned to the target_user_id of that selected target row.

48) gstat_reply_delay_mean   (minutes)
49) gstat_reply_delay_median (minutes)
50) gstat_reply_delay_std    (minutes; population std, ddof=0)
51) gstat_reply_delay_p05    (minutes; 5th percentile)
52) gstat_reply_delay_p95    (minutes; 95th percentile)
If an author has no valid delays, these are 0.0.

H. Optional engagement (if score present; scalar; z-scored)
Why: coarse audience response can correlate with style.

Scores are computed over target rows only.

53) gstat_score_mean
    Mean of score for target posts (non-numeric coerced to NaN).

54) gstat_score_var
    Variance (ddof=0) of score for target posts.

I. Sentence-level structure (scalar; z-scored)
Why: captures HOW users construct text at the sub-post level — sentence complexity,
rhetorical patterns, and syntactic elaboration. These are continuous, high-variance
signals that correlate with communication style and persona traits like
agreeableness (opener questions) and dominance (exclamation density).

Sentence splitting uses a lightweight regex heuristic (split on .!? followed by
space+uppercase or end-of-string).

55s) gstat_sent_per_post (per-post average)
    Mean number of sentences per post. Measures verbosity at the sentence level.

56s) gstat_sent_len_mean (corpus average; whitespace tokens per sentence)
    Mean sentence length in whitespace tokens across all sentences in the
    author's corpus. Longer sentences signal elaboration/complexity.

57s) gstat_sent_len_cv (corpus; coefficient of variation)
    std(sentence_lengths) / mean(sentence_lengths). Captures sentence length
    variability normalized by mean. High CV = mixed short/long sentences
    (rhetorical variation); low CV = uniform sentence construction.

58s) gstat_excl_sent_ratio (corpus ratio; per-sentence)
    Fraction of sentences ending with '!'. Measures exclamatory tendency
    at sentence granularity (vs. gstat_punct_ratio which counts characters).

59s) gstat_subordination_ratio (corpus ratio; per-word)
    Subordinating conjunction tokens / total words across all sentences.
    Higher values signal more complex, embedded clause structures.
    Lexicon: because, since, although, though, unless, while, whereas, if,
    when, whenever, wherever, after, before, until, as, that, which, who, whom, whose.

60s) gstat_opener_question_ratio (per-post average; boolean)
    Fraction of posts whose FIRST sentence ends with '?'. Measures the tendency
    to open with a question (engagement-seeking, Socratic style).

J. Post-level distributional statistics (scalar; z-scored)
Why: standard per-post ratios are averaged away to a single mean. These features
capture the SHAPE of the distribution across an author's posts — how consistent
or erratic the user's behavior is. Consistency is itself a persona signal.

61s) gstat_post_len_cv (per-user; coefficient of variation)
    std(post_lengths) / mean(post_lengths). High = unpredictable post length
    (alternates one-liners with walls of text); low = consistent length.

62s) gstat_post_len_skew (per-user; Fisher's skewness)
    Skewness of whitespace token counts across posts. Positive = mostly short
    posts with occasional long ones; negative = mostly long with occasional short.
    Clamped to [-10, 10] to avoid extreme outlier influence.

63s) gstat_style_volatility (per-user; std of per-post punct ratios)
    Standard deviation of per-post punctuation ratios. Measures how consistently
    a user uses punctuation — a proxy for emotional/stylistic stability.

64s) gstat_lexical_diversity_curve (per-user; TTR decay ratio)
    TTR(full_corpus) / TTR(half_corpus). Measures how quickly vocabulary
    exhausts across the author's posts. Low ratio = formulaic/repetitive
    writing (vocab plateaus early); high ratio ≈ 1.0 = sustained lexical diversity.

K. Response-context features (scalar; z-scored)
Why: how an author responds to context reveals communication style (elaboration,
engagement depth). These use both context and target rows in df_all.

65s) gstat_response_elaboration (per-user mean; ratio)
    Mean of (reply_word_count / context_word_count) across gids. Values >1.0
    indicate the author typically writes MORE than the preceding context;
    <1.0 indicates terse/minimal responses.

66s) gstat_ctx_len_preference (per-user mean; whitespace tokens)
    Mean total context length (in whitespace tokens) across the author's gids.
    Captures whether a user tends to engage in threads with long context (deep
    discussions) or short context (quick exchanges).

M. BEAST-GB meta-features (optional; derived from numeric globals)
Why: provide compact non-linear summaries of interactions among the global-static
signals without feeding raw text into the hypernetwork.

BEAST uses only the author table’s scalar globals as inputs. It does not read raw
text. It uses K-fold cross-validation over authors to produce out-of-fold (OOF)
predictions for every author.

Implementation details:
• Folds: BEAST_FOLDS env var (default 5), clamped to [2, #authors].
• Seed : BEAST_SEED  env var (default 142).
• Backend: XGBoost if importable, else sklearn GradientBoosting.
• Leaf embedding: tree leaf indices are hashed into a 64-dim normalized histogram.

Training tasks:

1) Sentiment regressor (OOF):
   y_sent := gstat_user_sent_mean
   X_sent := scalar globals excluding:
     • any gstat_personality* columns
     • any column containing "sent_" or "user_sent" or "gap_sentiment"
   Outputs:
   55) gstat_beast_sent_oof    : OOF prediction of y_sent
   56) gstat_beast_sent_resid  : y_sent - gstat_beast_sent_oof
   57) gstat_beast_sent_qrank  : percentile rank of gstat_beast_sent_oof over authors (0..1)
   61) gstat_beast_leaf64      : hashed leaf histogram embedding (list[64], sums to 1 when non-empty)

   Additionally, the script fits a full model on all authors to obtain feature
   importances and constructs:
   59) gstat_beast_comp_importance : importance-weighted composite score
       • Z = (X_sent - mean(X_sent)) / (std(X_sent) + 1e-6)
       • w = normalized feature importance weights
       • comp = (Z * w).sum(axis=1)
   60) gstat_beast_importance_conc : importance dispersion (constant across authors)
       • conc = 1 - ∑ w^2  (1 minus Herfindahl of importances)

2) Reactive classifier (OOF):
   Define a binary pseudo-label from reply delay:
   • delays := gstat_reply_delay_mean
   • thr := median(delays > 0) if any nonzero else median(delays)
   • y_reactive := 1 if delays <= thr else 0  (short-delay “reactive”)
   X_reactive := scalar globals excluding:
     • any gstat_personality* columns
     • any reply-delay columns (those containing "reply_delay")
   Output:
   58) gstat_beast_reactive_prob : OOF probability of y_reactive == 1

3) OCEAN-z regressors (OOF, one per trait):
   y_ocean := gstat_personality_z (5-dim; derived from gstat_personality_raw)
   X_ocean := scalar globals excluding any gstat_personality* columns
   Output:
   62) gstat_beast_ocean_pred : list[5] OOF predictions of OCEAN-z per author

BEAST robustness:
• If BEAST fails for any reason (missing deps, too few authors, etc.), all BEAST
  columns are emitted with zeros (and list outputs as all-zero vectors) to preserve
  schema continuity.

Normalization and datatypes
---------------------------
Scalar z-scoring:
• All scalar gstat_* columns (including BEAST scalars) are z-scored over AUTHORS
  (rows of author_static_10000.parquet):
    z = (x - μ) / (σ + 1e-6)
  where μ, σ are computed over authors (each author equally weighted).
• feature_norm_stats_10000.json stores μ and σ per scalar column.

Not scalar-z-scored:
• List-valued columns remain as lists of floats:
  gstat_personality_raw, gstat_personality_sigmoid_raw, gstat_personality_softmax_raw, gstat_personality_logits,
  gstat_personality_traits, gstat_personality_z,
  gstat_personality_sigmoid_traits, gstat_personality_sigmoid_z,
  gstat_personality_softmax_traits, gstat_personality_softmax_z,
  gstat_personality_logits_traits, gstat_personality_logits_z,
  gstat_hour_hist, gstat_beast_leaf64, gstat_beast_ocean_pred
• gstat_peak_hour remains an int32 (0..23).

Replication outputs:
• Per-thread global tables contain gid (int64), target_user_id (int64), and
  the same gstat_* columns (already normalized where applicable).

How these features are used
---------------------------
Given a thread gid, downstream training/inference retrieves its global-static
author vector g (from global_features_*.parquet). A hypernetwork consumes g and
produces a small parameter offset δθ that is added to the active PEFT module
(e.g., LoRA matrices) while the backbone stays frozen. This makes “persona switching”
a constant-time conditioning operation without swapping checkpoints.

Privacy note
------------
The builder derives compact statistics from anonymized text and coarse timestamps.
It does not emit usernames or raw message content in outputs; only numeric features.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from math import atan2, pi
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ────────────────────────── DEFAULT PATHS ────────────────────────── #

DEFAULT_DATA_DIR = Path("/workspace/hypernets/data")
DEFAULT_SENT_MODEL = Path("/workspace/hypernets/models/distilbert-sst2")
DEFAULT_PERSONA_MODEL = Path("/workspace/hypernets/models/Personality_LM")

# Corpus window length used for post-rate normalization (days in 2010-01-01..2016-01-01 inclusive)
CORPUS_DAYS = (pd.Timestamp("2016-01-01") - pd.Timestamp("2010-01-01")).days + 1

# Global seed for determinism across sampling / CV folds.
SEED = 142

# Rounding precision for list-valued features (for determinism)
ROUND_DP = 6


# ────────────────────────── LOGGING & ENV ────────────────────────── #

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("feature_builder")


# ────────────────────────── READABILITY FALLBACK ────────────────────────── #

try:
    import textstat as _textstat  # type: ignore

    def fk_grade(txt: str) -> float:
        try:
            return float(_textstat.flesch_kincaid_grade(txt)) if txt else 0.0
        except Exception:
            return 0.0

except Exception:
    _textstat = None  # type: ignore

    def _simple_syllables(word: str) -> int:
        word = (word or "").lower()
        vowels = "aeiouy"
        count, prev = 0, False
        for ch in word:
            v = ch in vowels
            if v and not prev:
                count += 1
            prev = v
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    def fk_grade(txt: str) -> float:
        if not txt or not txt.strip():
            return 0.0
        sents = [s for s in re.split(r"[.!?]+", txt) if s.strip()]
        words = re.findall(r"\w+", txt)
        n_s, n_w = max(1, len(sents)), max(1, len(words))
        sylls = sum(_simple_syllables(w) for w in words)
        return float(0.39 * (n_w / n_s) + 11.8 * (sylls / n_w) - 15.59)


# ────────────────────────── LEXICONS & PATTERNS ────────────────────────── #

_PROFANITY = {
    "fuck",
    "shit",
    "damn",
    "bitch",
    "bastard",
    "asshole",
    "dick",
    "crap",
    "piss",
    "darn",
    "bollocks",
    "bugger",
    "bloody",
    "shithead",
    "shitheads",
    "fucker",
    "motherfucker",
    "fucking",
    "cunt",
    "cunts",
    "cock",
    "prick",
    "whore",
    "slut",
    "douche",
    "douchebag",
    # LEXICON-REWRITE: morphological variants added for coverage
    "shitty",
    "shitting",
    "fucked",
    "fucks",
    "bitching",
    "crappy",
    "pissy",
    "dammit",
    "goddamn",
}
_ABUSE = {
    "idiot",
    "moron",
    "stupid",
    "loser",
    "dumb",
    "trash",
    "garbage",
    "scrub",
    "clown",
    "troll",
    "coward",
    "jerk",
    "pathetic",
    "retard",
    "retarded",
    "imbecile",
    "nitwit",
    "psycho",
    "weirdo",
    # LEXICON-REWRITE: expanded coverage for insult/abuse detection
    "fool",
    "ignorant",
    "incompetent",
    "worthless",
    "useless",
    "lazy",
    "delusional",
    "brainwashed",
    "nutjob",
    "creep",
    "freak",
}
_HEDGES_SINGLE = {
    "maybe",
    "perhaps",
    "probably",
    "possibly",
    "apparently",
    "seemingly",
    "arguably",
    "kinda",
    "sorta",
    "somewhat",
    "roughly",
    "presumably",
    "supposedly",
}
_HEDGES_MULTI = {
    "i think",
    "i guess",
    "i feel",
    "i believe",
    "kind of",
    "sort of",
    "it seems",
    "it appears",
}
_INTENSIFIERS = {
    "very",
    "really",
    "extremely",
    "absolutely",
    "totally",
    "completely",
    "utterly",
    "literally",
    "insanely",
    "incredibly",
    # LEXICON-REWRITE: removed "so" (primarily conjunction/discourse marker, not intensifier)
    "super",
    "highly",
    "seriously",
    "freaking",
    "hella",
}
# LEXICON-REWRITE: expanded with informal periphrastic modals (gonna/gotta/wanna/hafta/shall)
_MODALS = {"would", "could", "should", "might", "may", "can", "must", "ought",
           "gonna", "gotta", "wanna", "hafta", "shall"}
_NEGATIONS = {
    "not",
    "no",
    "never",
    "nothing",
    "nowhere",
    "nobody",
    "none",
    "isn't",
    "aren't",
    "don't",
    "doesn't",
    "can't",
    "won't",
    "shouldn't",
    "couldn't",
    "wouldn't",
    "didn't",
}
_SUBJECTIVE = {
    "amazing",
    "awful",
    "boring",
    "awesome",
    "terrible",
    "fantastic",
    "horrible",
    "lovely",
    "disgusting",
    "beautiful",
    "ridiculous",
    "adorable",
    "hate",
    "love",
    "prefer",
    "enjoy",
    # LEXICON-REWRITE: removed "like" (~80% filler on Reddit), "think"/"feel"/"believe"/
    # "seems"/"guess" (overlap with _HEDGES_MULTI)
    "dislike",
    "hope",
    "wish",
}
# LEXICON-REWRITE: removed "right" (too ambiguous: "right now", "right there"); added "definitely"
_AGREE = {"yes", "yeah", "yep", "correct", "true", "agree", "exactly", "indeed", "definitely"}
# LEXICON-REWRITE: multi-word agreement patterns (follows _HEDGES_MULTI approach)
_AGREE_MULTI = {
    "for sure", "no doubt", "spot on", "you're right", "i agree",
    "same here", "fair point", "good point", "well said",
}
# LEXICON-REWRITE: removed "no" (kept in _NEGATIONS), "false" (programming term);
# expanded single-word + added multi-word patterns for semantic disagreement
_DISAGREE = {
    "nope", "nah", "wrong", "disagree", "bs", "nonsense",
    "incorrect", "mistaken", "doubtful", "debatable",
    "questionable", "unlikely", "hardly",
}
_DISAGREE_MULTI = {
    "not true", "i disagree", "don't agree", "you're wrong",
    "not right", "absolutely not", "no way", "i doubt",
    "not really", "that's not",
}
# LEXICON-REWRITE: expanded with modern internet slang (ngl, fr, iirc, etc.)
_SLANG = {"lol", "lmao", "lmfao", "rofl", "idk", "imo", "imho", "btw", "tbh", "ikr", "afk", "brb", "smh", "fwiw",
           "ngl", "fr", "iirc", "eli5", "tbf", "rn", "irl", "nvm", "pls", "omg", "wtf"}
_EMOTICONS = {":)", ":-)", ":(", ":-(", ":D", ":-D", ":P", ":-P", ";)", "/:", ":/", "XD", "xD", ":|", ":-|"}

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

# LEXICON-REWRITE: removed duplicate " /s" (leading-space variant double-counts with "/s")
_SARC_MARKERS = {
    "/s",
    "yeah right",
    "sure jan",
    "as if",
    "totally not",
    "great...",
    "nice...",
    "right...",
    "ok...",
    "sure...",
}

# ── Sentence-level structure patterns ──
# Sentence boundary regex: splits on .!? followed by whitespace or end-of-string,
# but avoids splitting on abbreviations/decimals (simple heuristic).
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"\'])|(?<=[.!?])$')

# Subordinating conjunctions (signal complex sentence structure / elaboration).
_SUBORDINATORS = {
    "because", "since", "although", "though", "unless", "while", "whereas",
    "if", "when", "whenever", "wherever", "after", "before", "until",
    "as", "that", "which", "who", "whom", "whose",
}

# LEXICON-REWRITE: removed "no"/"not" (kept in _NEGATIONS), "so"/"very" (kept in _INTENSIFIERS),
# "can"/"should" (kept in _MODALS), "don" (not a real word; artifact of "don't" tokenization)
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "while",
    "as",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "to",
    "from",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "nor",
    "only",
    "own",
    "same",
    "than",
    "too",
    "will",
    "just",
    "now",
}

_FIRST_PERSON = {"i", "me", "my", "mine"}
_SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves", "u", "ya", "yall"}

# Unicode-aware word tokenization (letters only; keeps internal apostrophes).
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", flags=re.UNICODE)

# Contraction detection (counts occurrences; used only for a ratio feature).
_CONTRACTION_RE = re.compile(r"\b\w+['’](?:m|re|d|ll|ve|s|t)\b", re.IGNORECASE)

_ELLIPSIS_RE = re.compile(r"\.\.\.")
_PUNCT_BURST_RE = re.compile(r"([!?])\1+")


# ────────────────────────── LEXICON OVERLAP ASSERTION ────────────────────────── #
# LEXICON-REWRITE: dev safety net — each word should contribute to at most one feature ratio.
# This catches accidental re-introduction of cross-lexicon overlaps.
_DEDUP_LEXICONS = {
    "_STOPWORDS": _STOPWORDS,
    "_NEGATIONS": _NEGATIONS,
    "_INTENSIFIERS": _INTENSIFIERS,
    "_SUBJECTIVE": _SUBJECTIVE,
    "_AGREE": _AGREE,
    "_DISAGREE": _DISAGREE,
    "_MODALS": _MODALS,
    "_PROFANITY": _PROFANITY,
    "_ABUSE": _ABUSE,
    "_SLANG": _SLANG,
}
for _na, _sa in _DEDUP_LEXICONS.items():
    for _nb, _sb in _DEDUP_LEXICONS.items():
        if _na < _nb:
            _overlap = _sa & _sb
            if _overlap:
                raise RuntimeError(
                    f"Lexicon overlap detected: {_na} & {_nb} share {_overlap}. "
                    "Each word must belong to at most one feature lexicon."
                )
del _DEDUP_LEXICONS, _na, _sa, _nb, _sb, _overlap  # cleanup module namespace


# ────────────────────────── UTILITIES ────────────────────────── #


def _compute_corpus_days(df_tgt: pd.DataFrame) -> int:
    """Compute effective corpus window in days from actual data timestamps.

    LEXICON-REWRITE: replaces hardcoded CORPUS_DAYS (2010-2016) with data-derived value.
    Falls back to CORPUS_DAYS constant if timestamps are insufficient.
    """
    if "created_utc" not in df_tgt.columns:
        return CORPUS_DAYS
    ts = pd.to_datetime(df_tgt["created_utc"], errors="coerce").dropna()
    if len(ts) < 2:
        return CORPUS_DAYS
    span = (ts.max() - ts.min()).days + 1
    if span < 1:
        return CORPUS_DAYS
    return span


def round_list(x: Sequence[float], dp: int = ROUND_DP) -> List[float]:
    return [round(float(v), dp) for v in (list(x) if x is not None else [])]


def _is_target_mask(df: pd.DataFrame) -> pd.Series:
    if "group_label" not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    return df["group_label"].astype(str).str.endswith("_target")


def find_subreddit_column(df: pd.DataFrame) -> Optional[str]:
    for cand in (
        "subreddit",
        "subreddit_name",
        "subreddit_name_prefixed",
        "subreddit_id",
        "sr",
        "sr_name",
    ):
        if cand in df.columns:
            return cand
    return None


def toklist(text: str) -> List[str]:
    # casefold() is Unicode-robust; normalize curly apostrophes for lexicon consistency.
    s = (text or "").casefold().replace("’", "'")
    return _WORD_RE.findall(s)


def count_in_set(tokens: Sequence[str], vocab: set) -> int:
    return sum(1 for t in tokens if t in vocab)


def count_phrases(text: str, phrases: set) -> int:
    t = (text or "").casefold().replace("’", "'")
    return sum(t.count(p) for p in phrases)


def emoji_count(text: str) -> int:
    if not text:
        return 0
    # _EMOJI_RE matches runs; count codepoints inside the runs for a closer “count”.
    return int(sum(len(m) for m in _EMOJI_RE.findall(text)))


def emoticon_count(text: str) -> int:
    if not text:
        return 0
    t = text.replace("  ", " ")
    return sum(t.count(e) for e in _EMOTICONS)


def avg_word_len(tokens: Sequence[str]) -> float:
    return float(sum(len(t) for t in tokens) / max(1, len(tokens)))


def long_word_ratio(tokens: Sequence[str], n: int = 6) -> float:
    if not tokens:
        return 0.0
    return float(sum(1 for t in tokens if len(t) >= n) / len(tokens))


def stopword_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return float(sum(1 for t in tokens if t in _STOPWORDS) / len(tokens))


def punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    denom = max(1, len(text))

    # Exclude emoji so punctuation and emoji features do not double-count the same signal.
    emoji_chars = set("".join(_EMOJI_RE.findall(text)))
    punct = 0
    for ch in text:
        if ch in emoji_chars:
            continue
        if ch.isspace() or ch.isalnum():
            continue
        punct += 1
    return float(punct / denom)


def caps_ratio(text: str) -> float:
    toks = (text or "").split()
    if not toks:
        return 0.0
    return float(sum(1 for t in toks if t.isupper() and len(t) > 1) / len(toks))


def profanity_ratio(text: str) -> float:
    toks = toklist(text)
    if not toks:
        return 0.0
    return float(count_in_set(toks, _PROFANITY) / len(toks))


def firstperson_ratio(text: str) -> float:
    toks = toklist(text)
    if not toks:
        return 0.0
    return float(count_in_set(toks, _FIRST_PERSON) / len(toks))


def secondperson_ratio(text: str) -> float:
    toks = toklist(text)
    if not toks:
        return 0.0
    return float(count_in_set(toks, _SECOND_PERSON) / len(toks))


def link_flag(text: str) -> bool:
    t = text or ""
    if "<URL>" in t:
        return True
    return bool(re.search(r"(?i)\b(?:https?://|www\.)", t))


def contraction_ratio(texts: Sequence[str]) -> float:
    total_toks = sum(len(toklist(t)) for t in texts)
    if total_toks == 0:
        return 0.0
    matches = sum(len(_CONTRACTION_RE.findall(t)) for t in texts)
    return float(matches / total_toks)


def ellipsis_ratio(texts: Sequence[str]) -> float:
    chars = sum(len(t) for t in texts)
    if chars == 0:
        return 0.0
    return float(sum(len(_ELLIPSIS_RE.findall(t)) for t in texts) / chars)


def punct_burst_ratio(texts: Sequence[str]) -> float:
    chars = sum(len(t) for t in texts)
    if chars == 0:
        return 0.0
    return float(sum(len(_PUNCT_BURST_RE.findall(t)) for t in texts) / chars)


def sarcasm_marker_ratio(texts: Sequence[str]) -> float:
    """Fraction of posts containing at least one sarcasm marker (boolean per-post rate).

    LEXICON-REWRITE: converted from substring-count/token-count (units mismatch)
    to boolean per-post rate for dimensional consistency.
    """
    if not texts:
        return 0.0
    hits = 0
    for t in texts:
        tl = (t or "").casefold().replace("\u2019", "’")
        if any(tl.count(m) > 0 for m in _SARC_MARKERS):
            hits += 1
    return float(hits / len(texts))


def agreement_ratio(texts: Sequence[str]) -> float:
    """Agreement marker rate: single-word hits + multi-word phrase hits over token count.

    LEXICON-REWRITE: changed signature from tokens to texts to support multi-word
    patterns via _AGREE_MULTI (same approach as hedge_ratio).
    """
    tokens: List[str] = []
    for t in texts:
        tokens.extend(toklist(t))
    if not tokens:
        return 0.0
    single = count_in_set(tokens, _AGREE)
    multi = sum(count_phrases(t, _AGREE_MULTI) for t in texts)
    return float((single + multi) / len(tokens))


def disagreement_ratio(texts: Sequence[str]) -> float:
    """Disagreement marker rate: single-word hits + multi-word phrase hits over token count.

    LEXICON-REWRITE: changed signature from tokens to texts to support multi-word
    patterns via _DISAGREE_MULTI (same approach as hedge_ratio).
    """
    tokens: List[str] = []
    for t in texts:
        tokens.extend(toklist(t))
    if not tokens:
        return 0.0
    single = count_in_set(tokens, _DISAGREE)
    multi = sum(count_phrases(t, _DISAGREE_MULTI) for t in texts)
    return float((single + multi) / len(tokens))


def subjectivity_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return float(count_in_set(tokens, _SUBJECTIVE) / len(tokens))


def hedge_ratio(texts: Sequence[str]) -> float:
    tokens: List[str] = []
    for t in texts:
        tokens.extend(toklist(t))
    if not tokens:
        return 0.0
    single = count_in_set(tokens, _HEDGES_SINGLE)
    multi = sum(count_phrases(t, _HEDGES_MULTI) for t in texts)
    return float((single + multi) / len(tokens))


def intensifier_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return float(count_in_set(tokens, _INTENSIFIERS) / len(tokens))


def modal_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return float(count_in_set(tokens, _MODALS) / len(tokens))


def negation_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return float(count_in_set(tokens, _NEGATIONS) / len(tokens))


def slang_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return float(count_in_set(tokens, _SLANG) / len(tokens))


# ────────────────────────── SENTENCE & DISTRIBUTIONAL HELPERS ────────────────────────── #


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using a lightweight regex heuristic.

    Falls back to splitting on .!? boundaries. Returns at least one element
    (the full text) if no boundaries are found.
    """
    if not text or not text.strip():
        return []
    # Split on sentence-ending punctuation followed by space+uppercase or end-of-string.
    parts = _SENT_SPLIT_RE.split(text.strip())
    sents = [s.strip() for s in parts if s and s.strip()]
    return sents if sents else [text.strip()]


def _sent_level_features(texts: List[str]) -> Dict[str, float]:
    """Compute sentence-level structure features across an author's posts.

    Returns dict with keys: sent_per_post, sent_len_mean, sent_len_cv,
    excl_sent_ratio, subordination_ratio, opener_question_ratio.

    All values are continuous and high-variance by design.
    """
    if not texts:
        return {
            "sent_per_post": 0.0,
            "sent_len_mean": 0.0,
            "sent_len_cv": 0.0,
            "excl_sent_ratio": 0.0,
            "subordination_ratio": 0.0,
            "opener_question_ratio": 0.0,
        }

    all_sent_counts: List[int] = []
    all_sent_wlens: List[int] = []
    excl_sents = 0
    total_sents = 0
    sub_count = 0
    total_words_in_sents = 0
    opener_q = 0

    for t in texts:
        sents = split_sentences(t)
        n_sents = len(sents)
        all_sent_counts.append(n_sents)
        total_sents += n_sents

        for i, s in enumerate(sents):
            wlen = len(s.split())
            all_sent_wlens.append(wlen)

            # Exclamation sentence: ends with ! (after stripping whitespace)
            stripped = s.rstrip()
            if stripped.endswith("!"):
                excl_sents += 1

            # Opener question: first sentence of post ends with ?
            if i == 0 and stripped.endswith("?"):
                opener_q += 1

            # Subordination: count subordinating conjunctions in sentence
            stoks = set(s.casefold().split())
            sub_hits = len(stoks & _SUBORDINATORS)
            sub_count += sub_hits
            total_words_in_sents += wlen

    sent_per_post = float(np.mean(all_sent_counts)) if all_sent_counts else 0.0
    sent_len_mean = float(np.mean(all_sent_wlens)) if all_sent_wlens else 0.0
    # Coefficient of variation (std/mean) — captures sentence length variability
    # normalized by mean, so it's scale-free and comparable across verbose/terse users.
    if all_sent_wlens and sent_len_mean > 0:
        sent_len_cv = float(np.std(all_sent_wlens) / sent_len_mean)
    else:
        sent_len_cv = 0.0

    excl_sent_ratio = float(excl_sents / max(1, total_sents))
    subordination_ratio = float(sub_count / max(1, total_words_in_sents))
    opener_question_ratio = float(opener_q / max(1, len(texts)))

    return {
        "sent_per_post": sent_per_post,
        "sent_len_mean": sent_len_mean,
        "sent_len_cv": sent_len_cv,
        "excl_sent_ratio": excl_sent_ratio,
        "subordination_ratio": subordination_ratio,
        "opener_question_ratio": opener_question_ratio,
    }


def _post_distributional_features(
    texts: List[str], ws_counts: List[int], tokens_per_post: Optional[List[List[str]]] = None,
) -> Dict[str, float]:
    """Compute post-level distributional statistics across an author's posts.

    Returns dict with keys: post_len_cv, post_len_skew, style_volatility,
    lexical_diversity_curve.

    These capture HOW CONSISTENTLY a user writes, not just what they write.
    """
    if not texts or len(texts) < 2:
        return {
            "post_len_cv": 0.0,
            "post_len_skew": 0.0,
            "style_volatility": 0.0,
            "lexical_diversity_curve": 0.0,
        }

    # ── Post length coefficient of variation & skewness ──
    arr = np.array(ws_counts, dtype=np.float64)
    mu = arr.mean()
    std = arr.std()
    post_len_cv = float(std / max(mu, 1e-6))

    # Fisher's skewness: measures asymmetry of post length distribution
    # Positive skew = mostly short posts with occasional long ones
    # Negative skew = mostly long posts with occasional short ones
    if std > 1e-6 and len(arr) >= 3:
        n = len(arr)
        m3 = float(((arr - mu) ** 3).mean())
        post_len_skew = float(m3 / (std ** 3))
        # Clamp to avoid extreme outlier skewness values
        post_len_skew = max(-10.0, min(10.0, post_len_skew))
    else:
        post_len_skew = 0.0

    # ── Style volatility: variance of per-post punctuation ratios ──
    # Measures how consistent vs erratic punctuation usage is across posts.
    # A user who alternates between clean prose and punctuation-heavy rants
    # will have high volatility; a consistent user will have low.
    punct_per_post = [punct_ratio(t) for t in texts]
    style_volatility = float(np.std(punct_per_post)) if len(punct_per_post) >= 2 else 0.0

    # ── Lexical diversity curve (TTR decay rate) ──
    # Standard TTR conflates corpus size with diversity. Instead, we measure
    # how quickly new vocab is exhausted across the author's posts.
    # Compute TTR at 50% of corpus and compare to TTR at 100%.
    # A high ratio (close to 1.0) means vocabulary is rich and doesn't plateau early.
    # A low ratio means vocab exhausts quickly (formulaic/repetitive).
    if tokens_per_post and len(tokens_per_post) >= 2:
        # Accumulate vocab in document order
        all_toks: List[str] = []
        for tl in tokens_per_post:
            all_toks.extend(tl)
        total = len(all_toks)
        if total >= 4:
            mid = total // 2
            vocab_half = len(set(all_toks[:mid]))
            ttr_half = vocab_half / mid
            vocab_full = len(set(all_toks))
            ttr_full = vocab_full / total
            # Ratio of full-corpus TTR to half-corpus TTR
            # Low value = vocab exhaustion (repetitive); high value = sustained diversity
            lexical_diversity_curve = float(ttr_full / max(ttr_half, 1e-6))
        else:
            lexical_diversity_curve = 0.0
    else:
        lexical_diversity_curve = 0.0

    return {
        "post_len_cv": post_len_cv,
        "post_len_skew": post_len_skew,
        "style_volatility": style_volatility,
        "lexical_diversity_curve": lexical_diversity_curve,
    }


def compute_response_context_features(df_all: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """Pre-compute response-context features per target_user_id.

    Uses both context and target rows grouped by gid to measure how authors
    respond to context (elaboration ratio, context length preference).

    Returns: {uid: {"response_elaboration": float, "ctx_len_preference": float}}
    """
    result: Dict[int, Dict[str, float]] = {}
    required = {"gid", "target_user_id", "group_label", "text"}
    if not required.issubset(set(df_all.columns)):
        return result

    df = df_all[["gid", "target_user_id", "group_label", "text"]].copy()
    is_tgt = df["group_label"].astype(str).str.endswith("_target")

    # Target rows: compute reply length
    tgt = df.loc[is_tgt, ["gid", "target_user_id", "text"]].copy()
    tgt["reply_len"] = tgt["text"].astype(str).apply(lambda t: len(t.split()))
    # If multiple target rows per gid, take the first (shouldn't happen in normal data)
    tgt = tgt.drop_duplicates(subset=["gid"], keep="first")

    # Context rows: compute total context length per gid
    ctx = df.loc[~is_tgt, ["gid", "text"]].copy()
    ctx["ctx_len"] = ctx["text"].astype(str).apply(lambda t: len(t.split()))
    ctx_agg = ctx.groupby("gid", sort=False)["ctx_len"].sum().rename("ctx_total_len").reset_index()

    merged = tgt.merge(ctx_agg, on="gid", how="inner")
    if merged.empty:
        return result

    # Response elaboration: reply_len / ctx_total_len
    # >1.0 = user writes more than the context; <1.0 = terse responder
    merged["elab"] = merged["reply_len"] / merged["ctx_total_len"].clip(lower=1)

    # Aggregate per user
    by_uid = merged.groupby("target_user_id", sort=False).agg(
        elab_mean=("elab", "mean"),
        ctx_len_mean=("ctx_total_len", "mean"),
    )

    for uid, row in by_uid.iterrows():
        result[int(uid)] = {
            "response_elaboration": float(row["elab_mean"]),
            "ctx_len_preference": float(row["ctx_len_mean"]),
        }

    return result


# ────────────────────────── HF HELPERS ────────────────────────── #


def build_sentiment_pipe(model_path: str, device_id: int):
    """Return a HF pipeline for sentiment or None if unavailable."""
    try:
        from transformers import pipeline, logging as hf_log

        hf_log.set_verbosity_error()
        return pipeline(
            "sentiment-analysis",
            model=model_path,
            tokenizer=model_path,
            device=(device_id if device_id >= 0 else -1),
            return_all_scores=True,
            batch_size=128,
        )
    except Exception as e:
        LOGGER.warning("Sentiment pipeline unavailable (%s); falling back to zeros.", e)
        return None


@dataclass
class PersonalityModel:
    tokenizer: object
    model: object
    device: object
    idx_map: List[int]
    primary_activation: str


def build_personality_model(
    model_path: str,
    device_id: int,
    *,
    activation: str = "auto",
    idx_map_override: Optional[List[int]] = None,
) -> Optional[PersonalityModel]:
    canonical = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_path)
        mdl.eval()

        if device_id >= 0 and torch.cuda.is_available():
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        mdl.to(device)
    except Exception as e:
        LOGGER.warning("Personality model unavailable (%s); falling back to zeros.", e)
        return None

    activation = str(activation or "auto").strip().lower()
    if activation == "auto":
        problem_type = getattr(mdl.config, "problem_type", None)
        problem_type = str(problem_type).strip().lower() if problem_type is not None else ""

        if problem_type == "multi_label_classification":
            primary_activation = "sigmoid"
        elif problem_type in {"single_label_classification", "multi_class_classification"}:
            primary_activation = "softmax"
        elif problem_type == "regression":
            primary_activation = "regression"
        else:
            # Heuristics when config.problem_type is missing/ambiguous:
            # - num_labels == 1 strongly suggests regression
            # - otherwise default to sigmoid (safe for OCEAN-style independent traits)
            n_labels = getattr(mdl.config, "num_labels", None)
            if isinstance(n_labels, int) and n_labels == 1:
                primary_activation = "regression"
            else:
                primary_activation = "sigmoid"

            LOGGER.warning(
                "Personality model config.problem_type is unset/unknown (%r); defaulting primary_activation=%s. "
                "You can force a choice with --persona_activation sigmoid|softmax.",
                getattr(mdl.config, "problem_type", None),
                primary_activation,
            )
    else:
        if activation not in {"sigmoid", "softmax", "regression"}:
            raise ValueError(f"Invalid --persona_activation={activation!r}; expected auto|sigmoid|softmax")
        primary_activation = activation

    if idx_map_override is not None:
        if len(idx_map_override) != 5:
            raise ValueError("idx_map_override must have exactly 5 integers for [A,O,C,E,N]")
        idx_map = [int(x) for x in idx_map_override]
    else:
        id2label = getattr(mdl.config, "id2label", {}) or {}
        inv: Dict[str, int] = {}
        for i, lbl in id2label.items():
            try:
                inv[str(lbl).lower()] = int(i)
            except Exception:
                continue

        idx_map = []
        for name in canonical:
            idx = None
            for lbl, j in inv.items():
                if name == lbl or name in lbl:
                    idx = j
                    break
            if idx is None:
                try:
                    sorted_ids = sorted(int(k) for k in id2label.keys())
                    pos = canonical.index(name)
                    if pos < len(sorted_ids):
                        idx = sorted_ids[pos]
                except Exception:
                    idx = None
            if idx is None:
                idx = canonical.index(name)
            idx_map.append(int(idx))

    n_labels = getattr(mdl.config, "num_labels", None)
    if isinstance(n_labels, int) and n_labels > 0:
        idx_map = [min(max(0, i), n_labels - 1) for i in idx_map]

    LOGGER.info(
        "Personality model loaded: primary_activation=%s idx_map=%s num_labels=%s problem_type=%s",
        primary_activation,
        idx_map,
        getattr(mdl.config, "num_labels", None),
        getattr(mdl.config, "problem_type", None),
    )

    return PersonalityModel(tokenizer=tok, model=mdl, device=device, idx_map=idx_map, primary_activation=primary_activation)

# ────────────────────────── SENTIMENT CACHE ────────────────────────── #

POLAR_CACHE: "OrderedDict[str, float]" = OrderedDict()
try:
    POLAR_CACHE_MAX = int(os.getenv("POLAR_CACHE_MAX", "500000"))
except Exception:
    POLAR_CACHE_MAX = 500_000


def preprocess_sent(txt: str, max_toks: int = 256) -> str:
    toks = (txt or "").split()
    return " ".join(toks[:max_toks])


def polar_from_pipe(pipe, batch: Sequence[str]) -> List[float]:
    if pipe is None:
        return [0.0] * len(batch)
    outs = pipe(list(batch), truncation=True, max_length=256, padding=True)
    res: List[float] = []
    for s in outs:
        try:
            p_pos = next(d["score"] for d in s if str(d["label"]).upper().startswith("POS"))
            p_neg = next(d["score"] for d in s if str(d["label"]).upper().startswith("NEG"))
            res.append(float(p_pos - p_neg))
        except Exception:
            res.append(0.0)
    return res


def polarity_batch(pipe, texts: Sequence[str]) -> List[float]:
    """
    Compute sentiment polarity for each input text.

    Cache key uses the *truncated* form that is actually scored, to:
      - reduce memory pressure (no full-text keys)
      - avoid recomputing identical scored inputs
    Disable caching by setting: POLAR_CACHE_MAX=0
    """
    if pipe is None:
        return [0.0] * len(texts)

    keys = [preprocess_sent(t) for t in texts]

    # No-cache mode (useful for memory-constrained runs)
    if POLAR_CACHE_MAX <= 0:
        out: List[float] = []
        for i in range(0, len(keys), 128):
            out.extend(polar_from_pipe(pipe, keys[i : i + 128]))
        return [float(x) for x in out]

    need: List[str] = []
    for k in keys:
        if k in POLAR_CACHE:
            POLAR_CACHE.move_to_end(k)
        else:
            need.append(k)

    # Deduplicate misses while preserving order
    seen = set()
    need_uniq: List[str] = []
    for k in need:
        if k not in seen:
            need_uniq.append(k)
            seen.add(k)

    for i in range(0, len(need_uniq), 128):
        sub = need_uniq[i : i + 128]
        scores = polar_from_pipe(pipe, sub)
        for k, sc in zip(sub, scores):
            POLAR_CACHE[k] = float(sc)
            POLAR_CACHE.move_to_end(k)

    while len(POLAR_CACHE) > POLAR_CACHE_MAX:
        POLAR_CACHE.popitem(last=False)

    return [float(POLAR_CACHE.get(k, 0.0)) for k in keys]



# ────────────────────────── AUTHOR AGGREGATION HELPERS ────────────────────────── #


def compute_hour_hist_from_targets(df_tgt: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Per-user 24-bin hour histogram computed ONLY from _target rows."""
    hour_hist: Dict[int, np.ndarray] = {}
    if "created_utc" not in df_tgt.columns or "target_user_id" not in df_tgt.columns:
        return hour_hist

    # Robust for naive + tz-aware input; no np.issubdtype() on tz-aware dtypes.
    ts = pd.to_datetime(df_tgt["created_utc"], errors="coerce", utc=True)
    uid_ser = pd.to_numeric(df_tgt["target_user_id"], errors="coerce").dropna().astype("int64")
    uids = uid_ser.unique().tolist()

    df2 = pd.DataFrame(
        {
            "target_user_id": pd.to_numeric(df_tgt["target_user_id"], errors="coerce"),
            "created_utc": ts,
        }
    ).dropna(subset=["target_user_id"])

    df2["target_user_id"] = df2["target_user_id"].astype("int64")
    df2 = df2.dropna(subset=["created_utc"])

    if df2.empty:
        for uid in uids:
            hour_hist[int(uid)] = np.zeros(24, dtype=np.float32)
        return hour_hist

    df2["hour"] = df2["created_utc"].dt.hour.astype("int32")

    tbl = df2.groupby(["target_user_id", "hour"], sort=False).size().unstack(fill_value=0)

    # Guarantee 24 columns in order 0..23
    for h in range(24):
        if h not in tbl.columns:
            tbl[h] = 0
    tbl = tbl.reindex(sorted(tbl.columns), axis=1)

    for uid in uids:
        if uid in tbl.index:
            hour_hist[int(uid)] = tbl.loc[uid].to_numpy(dtype=np.float32)
        else:
            hour_hist[int(uid)] = np.zeros(24, dtype=np.float32)

    return hour_hist



def compute_reply_delays_from_all_rows(df_all: pd.DataFrame) -> Dict[int, List[float]]:
    """Reply delays (minutes) computed from full rows in a split.

    For each gid, delay = target_time - last_context_time.
    Assigned to the target_user_id from the _target row.

    Vectorized implementation:
      - deterministic selection if a gid ever has multiple target rows (take latest created_utc)
      - much faster than Python looping over every gid
    """
    delays: Dict[int, List[float]] = defaultdict(list)
    required = {"gid", "target_user_id", "group_label", "created_utc"}
    if not required.issubset(set(df_all.columns)):
        return delays

    df = df_all[["gid", "target_user_id", "group_label", "created_utc"]].copy()
    df["gid"] = pd.to_numeric(df["gid"], errors="coerce")
    df["target_user_id"] = pd.to_numeric(df["target_user_id"], errors="coerce")
    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["gid", "target_user_id"])
    df["gid"] = df["gid"].astype("int64")
    df["target_user_id"] = df["target_user_id"].astype("int64")

    is_tgt = df["group_label"].astype(str).str.endswith("_target")

    tgt = df.loc[is_tgt, ["gid", "target_user_id", "created_utc"]].dropna(subset=["created_utc"])
    if tgt.empty:
        return delays

    # If duplicates exist, choose latest target timestamp deterministically.
    tgt = tgt.sort_values(["gid", "created_utc"], kind="mergesort").drop_duplicates(subset=["gid"], keep="last")

    ctx = df.loc[~is_tgt, ["gid", "created_utc"]].dropna(subset=["created_utc"])
    if ctx.empty:
        return delays

    ctx_max = ctx.groupby("gid", sort=False)["created_utc"].max().rename("ctx_time").reset_index()

    merged = tgt.merge(ctx_max, on="gid", how="left")
    merged = merged.dropna(subset=["ctx_time"])
    if merged.empty:
        return delays

    delta_min = (merged["created_utc"] - merged["ctx_time"]).dt.total_seconds() / 60.0
    merged["delay_min"] = delta_min.astype("float32")

    merged = merged[np.isfinite(merged["delay_min"]) & (merged["delay_min"] >= 0)]
    if merged.empty:
        return delays

    by_uid = merged.groupby("target_user_id", sort=False)["delay_min"].apply(lambda s: s.astype("float32").tolist())
    for uid, lst in by_uid.items():
        delays[int(uid)] = list(lst)

    return delays



def compute_subreddit_concentration(df_tgt: pd.DataFrame) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Entropy/HHI/max-share over subreddits using ONLY target rows."""
    sr_col = find_subreddit_column(df_tgt)
    if sr_col is None:
        return {}, {}, {}

    sr_entropy: Dict[int, float] = {}
    sr_hhi: Dict[int, float] = {}
    sr_max: Dict[int, float] = {}

    # Cleaner casts subreddit to str; missing values can become "nan".
    missing_vals = {"", "nan", "none", "null"}

    for uid, grp in df_tgt.groupby("target_user_id", sort=False):
        vals = grp[sr_col].astype(str).str.strip()
        vals_l = vals.str.lower()
        vals = vals[(vals != "") & (~vals_l.isin(missing_vals))]

        counts = vals.value_counts(normalize=True)
        p = counts.values.astype(np.float64)

        if p.size == 0:
            sr_entropy[int(uid)] = 0.0
            sr_hhi[int(uid)] = 0.0
            sr_max[int(uid)] = 0.0
            continue

        ent = float(-(p * np.log2(p + 1e-12)).sum())
        hhi = float((p**2).sum())
        mx = float(p.max())
        sr_entropy[int(uid)] = ent
        sr_hhi[int(uid)] = hhi
        sr_max[int(uid)] = mx

    return sr_entropy, sr_hhi, sr_max



def circadian_derivatives(hist: np.ndarray) -> Tuple[List[float], float, float, int, float]:
    """Return (normalized 24vec, entropy, nocturnal_ratio, peak_hour, circular_mean_hour)."""
    if hist.sum() <= 0:
        vec = np.zeros(24, dtype=np.float32)
    else:
        vec = (hist / hist.sum()).astype(np.float32)

    p = vec + 1e-12
    entropy = float(-(p * np.log2(p)).sum())
    noct = float(vec[:6].sum())
    peak = int(vec.argmax())

    if vec.sum() == 0:
        cmean = 0.0
    else:
        angles = np.arange(24) * (2 * pi / 24)
        x = float((vec * np.cos(angles)).sum())
        y = float((vec * np.sin(angles)).sum())
        ang = atan2(y, x)
        if ang < 0:
            ang += 2 * pi
        cmean = float(ang * 24 / (2 * pi))
    return round_list(vec.tolist()), entropy, noct, peak, cmean


def det_sample_posts(grp: pd.DataFrame, max_n: int = 200) -> List[str]:
    """Deterministic sampling of posts for expensive model scoring."""
    cols = []
    if "created_utc" in grp.columns:
        cols.append("created_utc")
    if "gid" in grp.columns:
        cols.append("gid")
    if cols:
        grp = grp.sort_values(cols, kind="mergesort")
    return grp["text"].astype(str).head(max_n).tolist()

def compute_persona_for_user(
    posts: Sequence[str],
    persona: Optional[PersonalityModel],
    max_length: int = 64,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute personality summaries from posts.

    Returns:
        (primary_raw[5], sigmoid_raw[5], softmax_raw[5], mean_logits[5])

    Notes:
      - sigmoid_raw: independent sigmoid(logits) per trait
      - softmax_raw: softmax over the 5 selected trait dimensions (sums to 1)
      - mean_logits: regression-friendly mean of selected pre-activation logits
      - primary_raw: chosen by persona.primary_activation (sigmoid|softmax|regression)
    """
    if not posts or persona is None:
        z = [0.0] * 5
        return z, z, z, z

    try:
        import torch

        tok = persona.tokenizer
        mdl = persona.model
        device = persona.device
        idx = persona.idx_map

        sig_sum = torch.zeros(5, dtype=torch.float32)
        soft_sum = torch.zeros(5, dtype=torch.float32)
        logit_sum = torch.zeros(5, dtype=torch.float32)
        n = 0

        for i in range(0, len(posts), 128):
            batch = list(posts[i : i + 128])
            enc = tok(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = mdl(**enc)
                logits = out.logits
                logits_sel = logits[:, idx]

                probs_sig = torch.sigmoid(logits_sel)
                probs_soft = torch.softmax(logits_sel, dim=-1)

                sig_sum += probs_sig.sum(dim=0).detach().cpu()
                soft_sum += probs_soft.sum(dim=0).detach().cpu()
                logit_sum += logits_sel.sum(dim=0).detach().cpu()
                n += int(logits_sel.shape[0])

        n = max(1, n)
        sig_mean = (sig_sum / n).tolist()
        soft_mean = (soft_sum / n).tolist()
        logit_mean = (logit_sum / n).tolist()

        mode = str(getattr(persona, "primary_activation", "sigmoid")).strip().lower()
        if mode == "softmax":
            primary = soft_mean
        elif mode == "regression":
            primary = logit_mean
        else:
            primary = sig_mean

        return round_list(primary), round_list(sig_mean), round_list(soft_mean), round_list(logit_mean)
    except Exception as e:
        LOGGER.warning("Personality scoring failed (%s); returning zeros.", e)
        z = [0.0] * 5
        return z, z, z, z

# ────────────────────────── CONFIG ────────────────────────── #

@dataclass(frozen=True)
class Paths:
    train_parquet: Path
    val_parquet: Path
    test_parquet: Path
    output_dir: Path
    sent_model: Path
    persona_model: Path

    @property
    def author_parquet(self) -> Path:
        return self.output_dir / "author_static_10000.parquet"

    @property
    def norm_json(self) -> Path:
        return self.output_dir / "feature_norm_stats_10000.json"

    # ---- Combined (train+val+test gids; always written) ----
    @property
    def global_features_parquet(self) -> Path:
        return self.output_dir / "global_features_10000.parquet"

    @property
    def global_features_cols_json(self) -> Path:
        return self.output_dir / "global_features_10000_cols.json"

    # ---- Split-specific (optional via --split_outputs) ----
    @property
    def global_features_train_parquet(self) -> Path:
        return self.output_dir / "global_features_train_10000.parquet"

    @property
    def global_features_train_cols_json(self) -> Path:
        return self.output_dir / "global_features_train_10000_cols.json"

    @property
    def global_features_val_parquet(self) -> Path:
        return self.output_dir / "global_features_val_10000.parquet"

    @property
    def global_features_val_cols_json(self) -> Path:
        return self.output_dir / "global_features_val_10000_cols.json"

    @property
    def global_features_test_parquet(self) -> Path:
        return self.output_dir / "global_features_test_10000.parquet"

    @property
    def global_features_test_cols_json(self) -> Path:
        return self.output_dir / "global_features_test_10000_cols.json"

def read_split(path: Path, required: Sequence[str], optional: Sequence[str]) -> pd.DataFrame:
    dfp = pd.read_parquet(path)
    keep = [c for c in list(required) + list(optional) if c in dfp.columns]
    return dfp[keep]


def coerce_basic_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce key columns into stable dtypes.

    Notes:
      - created_utc is always coerced to timezone-aware UTC.
      - gid and target_user_id must be positive; invalid rows are dropped (and logged)
        instead of being silently mapped to 0 (which pollutes aggregates).
    """
    if "target_user_id" in df.columns:
        df["target_user_id"] = pd.to_numeric(df["target_user_id"], errors="coerce")
    if "gid" in df.columns:
        df["gid"] = pd.to_numeric(df["gid"], errors="coerce")

    invalid = pd.Series(False, index=df.index)
    if "target_user_id" in df.columns:
        invalid |= df["target_user_id"].isna() | (df["target_user_id"] <= 0)
    if "gid" in df.columns:
        invalid |= df["gid"].isna() | (df["gid"] <= 0)

    if bool(invalid.any()):
        n_bad = int(invalid.sum())
        LOGGER.warning("Dropping %d rows with invalid gid/target_user_id.", n_bad)
        df = df.loc[~invalid].copy()

    if "target_user_id" in df.columns:
        df["target_user_id"] = df["target_user_id"].astype("int64")
    if "gid" in df.columns:
        df["gid"] = df["gid"].astype("int64")

    if "created_utc" in df.columns:
        # Robust for naive and tz-aware inputs; always produces UTC tz-aware output.
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce", utc=True)

    if "group_label" not in df.columns:
        df["group_label"] = ""
    return df



def _read_unique_gids(path: Path) -> np.ndarray:
    df = pd.read_parquet(path, columns=["gid"])
    gids = pd.to_numeric(df["gid"], errors="coerce").fillna(0).astype("int64").to_numpy()
    gids = np.unique(gids)
    return gids


def verify_split_disjointness(paths: Paths) -> None:
    tr = _read_unique_gids(paths.train_parquet)
    va = _read_unique_gids(paths.val_parquet)
    te = _read_unique_gids(paths.test_parquet)

    tr_set = set(tr.tolist())
    va_set = set(va.tolist())
    te_set = set(te.tolist())

    tr_va = tr_set.intersection(va_set)
    tr_te = tr_set.intersection(te_set)
    va_te = va_set.intersection(te_set)

    if tr_va or tr_te or va_te:
        msg = []
        if tr_va:
            msg.append(f"train∩validation gids: {len(tr_va)}")
        if tr_te:
            msg.append(f"train∩test gids: {len(tr_te)}")
        if va_te:
            msg.append(f"validation∩test gids: {len(va_te)}")
        raise RuntimeError("Split leakage detected (gid overlap): " + ", ".join(msg))

    LOGGER.info("Split integrity ok: train/validation/test gid sets are disjoint.")


def _require_target_rows(df_all: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
    if "group_label" not in df_all.columns:
        raise RuntimeError(f"{split_name}: missing group_label column; cannot enforce _target semantics.")
    m_tgt = _is_target_mask(df_all)
    if not bool(m_tgt.any()):
        raise RuntimeError(f"{split_name}: no _target rows found.")
    return df_all.loc[m_tgt].copy()


# ────────────────────────── STAGE A: AUTHOR TABLE ────────────────────────── #


def build_author_table(
    paths: Paths,
    *,
    sent_device_id: int,
    persona_device_id: int,
    force: bool = False,
    disable_beast: bool = False,
    agg_split: str = "train",
    persona_activation: str = "auto",
    persona_idx_map: str = "",
) -> pd.DataFrame:
    out_path = paths.author_parquet
    if out_path.exists() and not force:
        LOGGER.info("Author table already exists → %s (use --force to rebuild)", out_path)
        return pd.read_parquet(out_path)

    if agg_split != "train":
        LOGGER.warning("agg_split=%s is not leakage-safe; evaluation splits may leak into author vectors.", agg_split)

    t0 = time.time()
    required_cols = ["gid", "target_user_id", "text", "created_utc", "group_label"]
    optional_cols = ["score", "subreddit", "subreddit_name", "subreddit_name_prefixed", "subreddit_id", "sr", "sr_name"]

    split_paths: List[Path]
    if agg_split == "train":
        split_paths = [paths.train_parquet]
    elif agg_split == "trainval":
        split_paths = [paths.train_parquet, paths.val_parquet]
    else:
        split_paths = [paths.train_parquet, paths.val_parquet, paths.test_parquet]

    LOGGER.info("Loading %s for author aggregation …", agg_split)
    df_all = pd.concat([read_split(p, required_cols, optional_cols) for p in split_paths], ignore_index=True)
    df_all = coerce_basic_types(df_all)

    df_tgt = _require_target_rows(df_all, split_name=f"author_agg[{agg_split}]")

    # LEXICON-REWRITE: compute corpus window from data instead of hardcoded CORPUS_DAYS
    corpus_days = _compute_corpus_days(df_tgt)
    LOGGER.info("Corpus window: %d days (hardcoded fallback: %d)", corpus_days, CORPUS_DAYS)

    LOGGER.info("Computing subreddit concentration metrics (%s; target-only) …", agg_split)
    sr_entropy, sr_hhi, sr_max = compute_subreddit_concentration(df_tgt)

    LOGGER.info("Computing reply-delay (%s; gid-based) …", agg_split)
    delays = compute_reply_delays_from_all_rows(df_all)

    LOGGER.info("Computing response-context features (%s; gid-based) …", agg_split)
    resp_ctx = compute_response_context_features(df_all)

    LOGGER.info("Computing circadian histograms (%s; target-only) …", agg_split)
    hour_hists = compute_hour_hist_from_targets(df_tgt)

    LOGGER.info("Estimating global sentiment baseline (%s; target-only) …", agg_split)
    sent_pipe = build_sentiment_pipe(str(paths.sent_model), sent_device_id)
    sample_n = min(1000, len(df_tgt))
    sample_texts = df_tgt["text"].astype(str).sample(sample_n, random_state=SEED).tolist() if sample_n > 0 else []
    global_sent = float(np.mean(polarity_batch(sent_pipe, sample_texts))) if sample_texts else 0.0

    idx_override = None
    if str(persona_idx_map).strip():
        parts = [p.strip() for p in str(persona_idx_map).split(",") if p.strip()]
        idx_override = [int(p) for p in parts]

    persona = build_personality_model(
        str(paths.persona_model),
        persona_device_id,
        activation=str(persona_activation),
        idx_map_override=idx_override,
    )

    has_score = "score" in df_tgt.columns
    n_users = int(df_tgt["target_user_id"].nunique())
    LOGGER.info("Aggregating author features (%s; users=%d; target-only) …", agg_split, n_users)

    rows: List[Dict[str, object]] = []
    last_log = time.time()

    for idx, (uid, grp) in enumerate(df_tgt.groupby("target_user_id", sort=False), 1):
        uid = int(uid)
        texts = grp["text"].astype(str).tolist()

        # Tokenization views:
        #   - toklist(): lexicon-style word tokens (letters + apostrophes, casefolded)
        #   - whitespace split: matches the dataset text field as-is (used for length + TTR)
        tokens_all: List[str] = []
        tokens_per_post: List[List[str]] = []
        ws_counts: List[int] = []
        ws_total = 0
        ws_vocab: set[str] = set()

        for t in texts:
            post_toks = toklist(t)
            tokens_all.extend(post_toks)
            tokens_per_post.append(post_toks)

            ws_toks = (t or "").split()
            ws_counts.append(len(ws_toks))
            ws_total += len(ws_toks)
            ws_vocab.update(ws_toks)

        svals = polarity_batch(sent_pipe, texts) if texts else [0.0]
        sent_mean = float(np.mean(svals)) if svals else 0.0
        sent_var = float(np.var(svals)) if svals else 0.0

        # Spec-aligned:
        # - mean whitespace token count per post
        # - TTR over whitespace tokens across the author corpus
        len_mean = float(np.mean(ws_counts)) if ws_counts else 0.0
        ttr = float(len(ws_vocab) / max(1, ws_total)) if ws_total > 0 else 0.0

        post_rate = float(len(texts) / corpus_days)


        punct_r = float(np.mean([punct_ratio(t) for t in texts])) if texts else 0.0
        question_r = float(np.mean([(t or "").strip().endswith("?") for t in texts])) if texts else 0.0
        caps_r = float(np.mean([caps_ratio(t) for t in texts])) if texts else 0.0
        prof_r = float(np.mean([profanity_ratio(t) for t in texts])) if texts else 0.0
        first_r = float(np.mean([firstperson_ratio(t) for t in texts])) if texts else 0.0
        second_r = float(np.mean([secondperson_ratio(t) for t in texts])) if texts else 0.0
        read_fk = float(np.median([fk_grade(t) for t in texts])) if texts else 0.0

        weekend_r = 0.0
        if "created_utc" in grp.columns:
            weekend_r = float(np.mean([(ts.weekday() >= 5) if pd.notna(ts) else 0.0 for ts in grp["created_utc"]]))

        link_r = float(np.mean([link_flag(t) for t in texts])) if texts else 0.0

        abuse_r = float(count_in_set(tokens_all, _ABUSE) / max(1, len(tokens_all)))
        hedge_r = hedge_ratio(texts)
        intens_r = intensifier_ratio(tokens_all)
        modal_r = modal_ratio(tokens_all)
        neg_r = negation_ratio(tokens_all)
        subj_r = subjectivity_ratio(tokens_all)
        # LEXICON-REWRITE: agreement/disagreement now take texts (not tokens) for multi-word support
        agree_r = agreement_ratio(texts)
        disagree_r = disagreement_ratio(texts)
        slang_r = slang_ratio(tokens_all)
        emoji_r = float(sum(emoji_count(t) for t in texts) / max(1, len(tokens_all)))
        # LEXICON-REWRITE: emoticon denominator fixed from tokens → chars (char-level numerator needs char-level denom)
        total_chars = max(1, sum(len(t) for t in texts))
        emoticon_r = float(sum(emoticon_count(t) for t in texts) / total_chars)
        contr_r = contraction_ratio(texts)
        ellip_r = ellipsis_ratio(texts)
        pb_r = punct_burst_ratio(texts)
        sarc_r = sarcasm_marker_ratio(texts)
        quote_r = float(sum((t or "").count('"') + (t or "").count("'") for t in texts) / total_chars)

        avg_wlen = avg_word_len(tokens_all)
        long_w = long_word_ratio(tokens_all, n=6)
        stop_w = stopword_ratio(tokens_all)

        if len(tokens_all) >= 2:
            bigrams = list(zip(tokens_all[:-1], tokens_all[1:]))
            rep_bigram_r = 1.0 - (len(set(bigrams)) / max(1, len(bigrams)))
        else:
            rep_bigram_r = 0.0

        # ── New feature block: sentence-level structure ──
        sent_feats = _sent_level_features(texts)

        # ── New feature block: post-level distributional statistics ──
        dist_feats = _post_distributional_features(texts, ws_counts, tokens_per_post)

        # ── New feature block: response-context features ──
        rctx = resp_ctx.get(uid, {"response_elaboration": 0.0, "ctx_len_preference": 0.0})

        if has_score:
            s = pd.to_numeric(grp["score"], errors="coerce")
            score_mean = float(s.mean()) if len(s) else 0.0
            score_var = float(s.var(ddof=0)) if len(s) else 0.0
        else:
            score_mean = 0.0
            score_var = 0.0

        sr_ent_val = float(sr_entropy.get(uid, 0.0))
        sr_herf = float(sr_hhi.get(uid, 0.0))
        sr_maxs = float(sr_max.get(uid, 0.0))

        hist_vec, hour_ent, noct_ratio, peak_hour, circ_mean = circadian_derivatives(
            hour_hists.get(uid, np.zeros(24, dtype=np.float32))
        )

        dlist = delays.get(uid, [])
        if dlist:
            darr = np.array(dlist, dtype=np.float32)
            reply_delay_mean = float(darr.mean())
            reply_delay_median = float(np.median(darr))
            reply_delay_std = float(darr.std())
            reply_delay_p05 = float(np.percentile(darr, 5))
            reply_delay_p95 = float(np.percentile(darr, 95))
        else:
            reply_delay_mean = reply_delay_median = reply_delay_std = reply_delay_p05 = reply_delay_p95 = 0.0

        posts = det_sample_posts(grp, max_n=200)
        prob_primary, prob_sigmoid, prob_softmax, logits_mean = compute_persona_for_user(posts, persona)

        row: Dict[str, object] = {
            "target_user_id": uid,
            "gstat_personality_raw": round_list(prob_primary),
            "gstat_personality_sigmoid_raw": round_list(prob_sigmoid),
            "gstat_personality_softmax_raw": round_list(prob_softmax),
            "gstat_personality_logits": round_list(logits_mean),
            "gstat_gap_sentiment": round(float(sent_mean - global_sent), ROUND_DP),
            "gstat_user_sent_mean": round(sent_mean, ROUND_DP),
            "gstat_user_sent_var": round(sent_var, ROUND_DP),
            "gstat_user_len_mean": round(len_mean, ROUND_DP),
            "gstat_user_ttr": round(ttr, ROUND_DP),
            "gstat_user_post_rate": round(post_rate, ROUND_DP),
            "gstat_user_subreddit_entropy": round(sr_ent_val, ROUND_DP),
            "gstat_user_sr_herfindahl": round(sr_herf, ROUND_DP),
            "gstat_user_sr_max_share": round(sr_maxs, ROUND_DP),
            "gstat_punct_ratio": round(punct_r, ROUND_DP),
            "gstat_question_ratio": round(question_r, ROUND_DP),
            "gstat_caps_ratio": round(caps_r, ROUND_DP),
            "gstat_profanity_ratio": round(prof_r, ROUND_DP),
            "gstat_abuse_ratio": round(abuse_r, ROUND_DP),
            "gstat_firstperson_ratio": round(first_r, ROUND_DP),
            "gstat_secondperson_ratio": round(second_r, ROUND_DP),
            "gstat_readability_fk": round(read_fk, ROUND_DP),
            "gstat_weekend_ratio": round(weekend_r, ROUND_DP),
            "gstat_link_ratio": round(link_r, ROUND_DP),
            "gstat_hedge_ratio": round(hedge_r, ROUND_DP),
            "gstat_intensifier_ratio": round(intens_r, ROUND_DP),
            "gstat_modal_ratio": round(modal_r, ROUND_DP),
            "gstat_negation_ratio": round(neg_r, ROUND_DP),
            "gstat_subjectivity_ratio": round(subj_r, ROUND_DP),
            "gstat_agreement_ratio": round(agree_r, ROUND_DP),
            "gstat_disagreement_ratio": round(disagree_r, ROUND_DP),
            "gstat_slang_ratio": round(slang_r, ROUND_DP),
            "gstat_emoji_ratio": round(emoji_r, ROUND_DP),
            "gstat_emoticon_ratio": round(emoticon_r, ROUND_DP),
            "gstat_contraction_ratio": round(contr_r, ROUND_DP),
            "gstat_ellipsis_ratio": round(ellip_r, ROUND_DP),
            "gstat_punct_burst_ratio": round(pb_r, ROUND_DP),
            "gstat_sarcasm_ratio": round(sarc_r, ROUND_DP),
            "gstat_quote_ratio": round(quote_r, ROUND_DP),
            "gstat_avg_word_len": round(avg_wlen, ROUND_DP),
            "gstat_long_word_ratio": round(long_w, ROUND_DP),
            "gstat_stopword_ratio": round(stop_w, ROUND_DP),
            "gstat_rep_bigram_ratio": round(rep_bigram_r, ROUND_DP),
            "gstat_hour_hist": round_list(hist_vec),
            "gstat_hour_entropy": round(hour_ent, ROUND_DP),
            "gstat_nocturnal_ratio": round(noct_ratio, ROUND_DP),
            "gstat_peak_hour": int(peak_hour),
            "gstat_circadian_mean": round(circ_mean, ROUND_DP),
            "gstat_reply_delay_mean": round(reply_delay_mean, ROUND_DP),
            "gstat_reply_delay_median": round(reply_delay_median, ROUND_DP),
            "gstat_reply_delay_std": round(reply_delay_std, ROUND_DP),
            "gstat_reply_delay_p05": round(reply_delay_p05, ROUND_DP),
            "gstat_reply_delay_p95": round(reply_delay_p95, ROUND_DP),
            "gstat_score_mean": round(score_mean, ROUND_DP),
            "gstat_score_var": round(score_var, ROUND_DP),
            # ── Sentence-level structure features ──
            "gstat_sent_per_post": round(sent_feats["sent_per_post"], ROUND_DP),
            "gstat_sent_len_mean": round(sent_feats["sent_len_mean"], ROUND_DP),
            "gstat_sent_len_cv": round(sent_feats["sent_len_cv"], ROUND_DP),
            "gstat_excl_sent_ratio": round(sent_feats["excl_sent_ratio"], ROUND_DP),
            "gstat_subordination_ratio": round(sent_feats["subordination_ratio"], ROUND_DP),
            "gstat_opener_question_ratio": round(sent_feats["opener_question_ratio"], ROUND_DP),
            # ── Post-level distributional statistics ──
            "gstat_post_len_cv": round(dist_feats["post_len_cv"], ROUND_DP),
            "gstat_post_len_skew": round(dist_feats["post_len_skew"], ROUND_DP),
            "gstat_style_volatility": round(dist_feats["style_volatility"], ROUND_DP),
            "gstat_lexical_diversity_curve": round(dist_feats["lexical_diversity_curve"], ROUND_DP),
            # ── Response-context features ──
            "gstat_response_elaboration": round(rctx["response_elaboration"], ROUND_DP),
            "gstat_ctx_len_preference": round(rctx["ctx_len_preference"], ROUND_DP),
        }
        rows.append(row)

        if time.time() - last_log > 10.0:
            LOGGER.info("Authors processed: %d / %d", idx, n_users)
            last_log = time.time()

    adf = pd.DataFrame(rows).sort_values("target_user_id").reset_index(drop=True)

    if len(adf) > 0:
        def _traits_and_z(col: str) -> Tuple[List[List[float]], List[List[float]]]:
            mat = np.vstack(adf[col].tolist()).astype(np.float32)
            med = np.median(mat, axis=0).astype(np.float32)
            traits = (mat - med).astype(np.float32)

            mean_t = traits.mean(axis=0)
            std_t = traits.std(axis=0) + 1e-6
            z = (traits - mean_t) / std_t

            return [round_list(v.tolist()) for v in traits], [round_list(v.tolist()) for v in z]

        # Backward-compatible primary traits/z (whatever gstat_personality_raw contains)
        traits_p, z_p = _traits_and_z("gstat_personality_raw")
        adf["gstat_personality_traits"] = traits_p
        adf["gstat_personality_z"] = z_p

        # Always emit alternates so you can ablate without rebuilding features
        traits_s, z_s = _traits_and_z("gstat_personality_sigmoid_raw")
        adf["gstat_personality_sigmoid_traits"] = traits_s
        adf["gstat_personality_sigmoid_z"] = z_s

        traits_m, z_m = _traits_and_z("gstat_personality_softmax_raw")
        adf["gstat_personality_softmax_traits"] = traits_m
        adf["gstat_personality_softmax_z"] = z_m

        traits_l, z_l = _traits_and_z("gstat_personality_logits")
        adf["gstat_personality_logits_traits"] = traits_l
        adf["gstat_personality_logits_z"] = z_l
    else:
        adf["gstat_personality_traits"] = []
        adf["gstat_personality_z"] = []
        adf["gstat_personality_sigmoid_traits"] = []
        adf["gstat_personality_sigmoid_z"] = []
        adf["gstat_personality_softmax_traits"] = []
        adf["gstat_personality_softmax_z"] = []
        adf["gstat_personality_logits_traits"] = []
        adf["gstat_personality_logits_z"] = []

    if disable_beast:
        LOGGER.info("[BEAST] disabled via --disable_beast; emitting zeros for BEAST columns.")
        adf["gstat_beast_sent_oof"] = 0.0
        adf["gstat_beast_sent_resid"] = 0.0
        adf["gstat_beast_sent_qrank"] = 0.0
        adf["gstat_beast_reactive_prob"] = 0.0
        adf["gstat_beast_comp_importance"] = 0.0
        adf["gstat_beast_importance_conc"] = 0.0
        adf["gstat_beast_leaf64"] = [[0.0] * 64 for _ in range(len(adf))]
        adf["gstat_beast_ocean_pred"] = [[0.0] * 5 for _ in range(len(adf))]
    else:
        _add_beast_features(adf)

    list_cols_final = {
        "gstat_personality_raw",
        "gstat_personality_sigmoid_raw",
        "gstat_personality_softmax_raw",
        "gstat_personality_logits",
        "gstat_personality_traits",
        "gstat_personality_z",
        "gstat_personality_sigmoid_traits",
        "gstat_personality_sigmoid_z",
        "gstat_personality_softmax_traits",
        "gstat_personality_softmax_z",
        "gstat_personality_logits_traits",
        "gstat_personality_logits_z",
        "gstat_hour_hist",
        "gstat_beast_leaf64",
        "gstat_beast_ocean_pred",
    }
    skip_cols = {"target_user_id", "gstat_peak_hour"} | list_cols_final

    norm_stats: Dict[str, Dict[str, float]] = {}
    for c in adf.columns:
        if c in skip_cols:
            continue
        if adf[c].dtype == object:
            continue
        m = float(adf[c].mean())
        s = float(adf[c].std(ddof=0) + 1e-6)
        norm_stats[c] = {"mean": m, "std": s}
        adf[c] = (adf[c].astype("float32") - m) / s

    adf["target_user_id"] = adf["target_user_id"].astype("int64")
    if "gstat_peak_hour" in adf.columns:
        adf["gstat_peak_hour"] = adf["gstat_peak_hour"].astype("int32")

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    adf.to_parquet(out_path, index=False)
    with open(paths.norm_json, "w") as fh:
        json.dump(norm_stats, fh)

    LOGGER.info("Author table written: %s (rows=%d) [%.1fs]", out_path, len(adf), time.time() - t0)
    return adf


def _add_beast_features(adf: pd.DataFrame, leaf_dim: int = 64) -> None:
    try:
        list_cols = {
            "gstat_personality_raw",
            "gstat_personality_logits",
            "gstat_personality_traits",
            "gstat_personality_z",
            "gstat_hour_hist",
        }
        scalar_cols = [
            c
            for c in adf.columns
            if c.startswith("gstat_") and c not in list_cols and c not in {"target_user_id"} and adf[c].dtype != object
        ]

        X_cols_sent = [
            c
            for c in scalar_cols
            if (not c.startswith("gstat_personality"))
            and ("sent_" not in c and "user_sent" not in c and "gap_sentiment" not in c)
        ]
        X_sent = adf[X_cols_sent].to_numpy(dtype=np.float32)
        y_sent = adf["gstat_user_sent_mean"].to_numpy(dtype=np.float32)

        delay_cols = [c for c in scalar_cols if "reply_delay" in c]
        X_cols_reactive = [c for c in scalar_cols if (c not in delay_cols) and (not c.startswith("gstat_personality"))]
        X_reactive = adf[X_cols_reactive].to_numpy(dtype=np.float32)

        delays = adf["gstat_reply_delay_mean"].to_numpy(dtype=np.float32)
        nonzero = delays[delays > 0]
        thr = float(np.median(nonzero)) if nonzero.size > 0 else float(np.median(delays))
        y_reactive = (delays <= thr).astype(np.int32)

        X_cols_ocean = [c for c in scalar_cols if not c.startswith("gstat_personality")]
        X_ocean = adf[X_cols_ocean].to_numpy(dtype=np.float32)
        y_ocean = (
            np.vstack(adf["gstat_personality_z"].tolist()).astype(np.float32)
            if len(adf)
            else np.zeros((0, 5), dtype=np.float32)
        )

        def _try_import_xgb():
            try:
                import xgboost as xgb  # type: ignore

                return xgb
            except Exception:
                return None

        def _fit_gbdt_regressor(X: np.ndarray, y: np.ndarray, random_state: int = SEED):
            xgb = _try_import_xgb()
            if xgb is not None:
                params = dict(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    tree_method="hist",
                    n_jobs=int(os.getenv("CPU_THREADS", "8")),
                    random_state=random_state,
                )
                try:
                    model = xgb.XGBRegressor(**params)
                    model.fit(X, y)
                    return model, "xgb"
                except Exception:
                    pass

            from sklearn.ensemble import GradientBoostingRegressor  # type: ignore

            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                random_state=random_state,
            )
            model.fit(X, y)
            return model, "sk"

        def _fit_gbdt_classifier(X: np.ndarray, y: np.ndarray, random_state: int = SEED):
            xgb = _try_import_xgb()
            if xgb is not None:
                params = dict(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    tree_method="hist",
                    n_jobs=int(os.getenv("CPU_THREADS", "8")),
                    random_state=random_state,
                    eval_metric="logloss",
                )
                try:
                    model = xgb.XGBClassifier(**params)
                    model.fit(X, y)
                    return model, "xgb"
                except Exception:
                    pass

            from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                random_state=random_state,
            )
            model.fit(X, y)
            return model, "sk"

        def _gbdt_apply_leaves(model, backend: str, X: np.ndarray) -> np.ndarray:
            if model is None:
                return np.zeros((X.shape[0], 0), dtype=np.int32)
            if backend == "xgb":
                try:
                    leaves = model.apply(X)
                    if leaves.ndim == 1:
                        leaves = leaves.reshape(-1, 1)
                    return leaves.astype(np.int32)
                except Exception:
                    pass
            try:
                leaves = []
                for est in model.estimators_.ravel():
                    leaves.append(est.apply(X))
                return np.vstack(leaves).T.astype(np.int32)
            except Exception:
                return np.zeros((X.shape[0], 0), dtype=np.int32)

        from sklearn.model_selection import KFold  # type: ignore

        folds = int(os.getenv("BEAST_FOLDS", "5"))
        seed = int(os.getenv("BEAST_SEED", str(SEED)))

        N = int(X_sent.shape[0])
        if N < 2:
            raise RuntimeError("BEAST requires at least 2 authors")
        folds = max(2, min(int(folds), int(N)))

        oof = np.zeros(N, dtype=np.float32)
        leaves_store: List[Optional[np.ndarray]] = [None] * N
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

        for tr, va in kf.split(np.arange(N)):
            m, backend = _fit_gbdt_regressor(X_sent[tr], y_sent[tr], random_state=seed)
            oof[va] = m.predict(X_sent[va]).astype(np.float32)
            Lv = _gbdt_apply_leaves(m, backend=backend, X=X_sent[va])
            for j, rid in enumerate(va):
                leaves_store[rid] = Lv[j]

        def _hash_leaf_indices_to_vec(leaf_idx: np.ndarray, k: int = leaf_dim) -> np.ndarray:
            if leaf_idx is None or leaf_idx.size == 0:
                return np.zeros(k, dtype=np.float32)
            if leaf_idx.ndim > 1:
                leaf_idx = leaf_idx.ravel()
            out = np.zeros(k, dtype=np.float32)
            for t, lid in enumerate(leaf_idx.tolist()):
                h = (t * 1315423911 + int(lid)) % k
                out[h] += 1.0
            s = out.sum()
            return out / s if s > 0 else out

        leaf64 = np.vstack(
            [
                _hash_leaf_indices_to_vec(
                    np.asarray(leaves_store[i]) if leaves_store[i] is not None else np.asarray([]),
                    k=leaf_dim,
                )
                for i in range(N)
            ]
        )

        model_full, backend_full = _fit_gbdt_regressor(X_sent, y_sent, random_state=seed)

        imps: Dict[str, float] = {}
        if backend_full == "xgb":
            try:
                score = model_full.get_booster().get_score(importance_type="gain")  # type: ignore[attr-defined]
                for k, v in score.items():
                    if k.startswith("f"):
                        idx = int(k[1:])
                        if 0 <= idx < len(X_cols_sent):
                            imps[X_cols_sent[idx]] = float(v)
            except Exception:
                pass

        if not imps:
            vals = getattr(model_full, "feature_importances_", None)
            if vals is not None:
                vals = np.asarray(vals, dtype=np.float32)
                s = float(vals.sum()) or 1.0
                imps = {c: float(v / s) for c, v in zip(X_cols_sent, vals.tolist())}

        mu = X_sent.mean(axis=0)
        sd = X_sent.std(axis=0) + 1e-6
        Z = (X_sent - mu) / sd
        w = np.array([imps.get(c, 0.0) for c in X_cols_sent], dtype=np.float32)
        if w.sum() > 0:
            w = w / w.sum()
            comp = (Z * w).sum(axis=1).astype(np.float32)
            conc = float(1.0 - float((w**2).sum()))
        else:
            comp = np.zeros(N, dtype=np.float32)
            conc = 0.0

        resid = (y_sent - oof).astype(np.float32)
        qrank = pd.Series(oof).rank(method="average", pct=True).to_numpy(dtype=np.float32)

        reactive_oof = np.zeros(N, dtype=np.float32)
        for tr, va in kf.split(np.arange(N)):
            m, backend = _fit_gbdt_classifier(X_reactive[tr], y_reactive[tr], random_state=seed)
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X_reactive[va])[:, 1]
            else:
                p = m.predict(X_reactive[va])
            reactive_oof[va] = np.asarray(p, dtype=np.float32)

        ocean_oof = np.zeros((N, 5), dtype=np.float32)
        for t_idx in range(5):
            y_t = y_ocean[:, t_idx].astype(np.float32)
            for tr, va in kf.split(np.arange(N)):
                m, backend = _fit_gbdt_regressor(X_ocean[tr], y_t[tr], random_state=seed + 11 + t_idx)
                ocean_oof[va, t_idx] = m.predict(X_ocean[va]).astype(np.float32)

        adf["gstat_beast_sent_oof"] = oof.astype(np.float32)
        adf["gstat_beast_sent_resid"] = resid
        adf["gstat_beast_sent_qrank"] = qrank
        adf["gstat_beast_reactive_prob"] = reactive_oof
        adf["gstat_beast_comp_importance"] = comp
        adf["gstat_beast_importance_conc"] = np.full(N, conc, dtype=np.float32)
        adf["gstat_beast_leaf64"] = [round_list(leaf64[i].tolist()) for i in range(N)]
        adf["gstat_beast_ocean_pred"] = [round_list(ocean_oof[i].tolist()) for i in range(N)]

        LOGGER.info("[BEAST] Added meta-features (sent_oof/resid/qrank/reactive_prob/leaf64/comp/conc/ocean_pred).")
    except Exception as e:
        LOGGER.warning("[BEAST] Skipping meta-features due to error: %s", e)
        N = len(adf)
        adf["gstat_beast_sent_oof"] = 0.0
        adf["gstat_beast_sent_resid"] = 0.0
        adf["gstat_beast_sent_qrank"] = 0.0
        adf["gstat_beast_reactive_prob"] = 0.0
        adf["gstat_beast_comp_importance"] = 0.0
        adf["gstat_beast_importance_conc"] = 0.0
        adf["gstat_beast_leaf64"] = [[0.0] * leaf_dim for _ in range(N)]
        adf["gstat_beast_ocean_pred"] = [[0.0] * 5 for _ in range(N)]


# ────────────────────────── STAGE B: REPLICATION TO PER-GID ────────────────────────── #


def _build_default_feature_row(adf: pd.DataFrame) -> Dict[str, object]:
    gstat_cols = [c for c in adf.columns if c.startswith("gstat_")]
    defaults: Dict[str, object] = {}
    for c in gstat_cols:
        v = adf[c].iloc[0] if len(adf) else 0.0
        if isinstance(v, (list, tuple, np.ndarray)):
            L = len(v)
            defaults[c] = [0.0] * L
        elif pd.api.types.is_integer_dtype(adf[c]):
            defaults[c] = 0
        else:
            defaults[c] = 0.0
    return defaults


def replicate_to_global_parts(
    *,
    rank: int,
    world_size: int,
    paths: Paths,
    out_base: Path,
    split_paths: Sequence[Path],
    resume: bool = False,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_part = out_base.with_name(f"{out_base.stem}_w{rank}{out_base.suffix}")
    ckpt_path = paths.output_dir / f"feature_builder_ckpt_{out_base.stem}_{rank}.json"

    adf = pd.read_parquet(paths.author_parquet).reset_index(drop=True)
    if len(adf) == 0:
        raise RuntimeError(f"Author table is empty: {paths.author_parquet}")

    uid_to_row = {int(uid): i for i, uid in enumerate(adf["target_user_id"].tolist())}
    gstat_cols = [c for c in adf.columns if c.startswith("gstat_")]
    list_like_cols = {c for c in gstat_cols if adf[c].dtype == object}
    default_feats = _build_default_feature_row(adf)

    # Freeze Parquet schema to avoid per-chunk inference mismatch.
    fields = [
        pa.field("gid", pa.int64()),
        pa.field("target_user_id", pa.int64()),
    ]
    for c in gstat_cols:
        if c in list_like_cols:
            fields.append(pa.field(c, pa.list_(pa.float32())))
        elif pd.api.types.is_integer_dtype(adf[c]):
            if str(adf[c].dtype) == "int32":
                fields.append(pa.field(c, pa.int32()))
            else:
                fields.append(pa.field(c, pa.int64()))
        else:
            fields.append(pa.field(c, pa.float32()))
    schema = pa.schema(fields)

    cols = ["gid", "target_user_id", "group_label"]

    def _read_min(path: Path) -> pd.DataFrame:
        # Read only the columns needed for replication (avoids pulling full text into memory).
        try:
            return pd.read_parquet(path, columns=cols)
        except Exception:
            dfp = pd.read_parquet(path)
            keep = [c for c in cols if c in dfp.columns]
            return dfp[keep]


    df = pd.concat([_read_min(p) for p in split_paths], ignore_index=True)
    df = coerce_basic_types(df)

    gold_raw = _require_target_rows(df, split_name=out_base.name)
    if gold_raw["gid"].duplicated(keep=False).any():
        dup = int(gold_raw["gid"].duplicated(keep=False).sum())
        raise RuntimeError(f"{out_base.name}: found multiple _target rows for some gids (rows involved={dup}).")

    gold = gold_raw.sort_values("gid", kind="mergesort").reset_index(drop=True)
    gold = gold[gold["gid"] % world_size == rank].reset_index(drop=True)
    LOGGER.info("%s rank %d: %d gid rows selected for replication.", out_base.name, rank, len(gold))

    last_gid = -1
    if resume and ckpt_path.exists():
        try:
            last_gid = int(json.load(open(ckpt_path))["last_gid"])
            LOGGER.info("%s rank %d: resume from last_gid=%d", out_base.name, rank, last_gid)
        except Exception:
            last_gid = -1

    def _save_ckpt(val: int) -> None:
        try:
            paths.output_dir.mkdir(parents=True, exist_ok=True)
            with open(ckpt_path, "w") as fh:
                json.dump({"last_gid": int(val)}, fh)
        except Exception:
            pass

    writer: Optional[pq.ParquetWriter] = None
    chunk: List[Dict[str, object]] = []
    CHUNK_SIZE = 200_000

    missing_uids = 0
    t0 = time.time()

    for n, row in enumerate(gold.itertuples(index=False), 1):
        gid = int(row.gid)
        if gid <= last_gid:
            continue
        uid = int(row.target_user_id)
        ridx = uid_to_row.get(uid)

        out: Dict[str, object] = {"gid": gid, "target_user_id": uid}

        if ridx is None:
            missing_uids += 1
            for c in gstat_cols:
                out[c] = default_feats[c]
        else:
            arow = adf.iloc[ridx]
            for c in gstat_cols:
                v = arow[c]
                if c in list_like_cols and isinstance(v, np.ndarray):
                    v = v.tolist()
                out[c] = v

        chunk.append(out)

        if len(chunk) >= CHUNK_SIZE:
            tbl = pa.Table.from_pylist(chunk, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(out_part, schema=schema, compression="snappy")
            writer.write_table(tbl)
            chunk.clear()
            _save_ckpt(gid)

            if len(gold) > 0:
                LOGGER.info(
                    "%s rank %d progress: %d / %d rows written (%.1f%%)",
                    out_base.name,
                    rank,
                    n,
                    len(gold),
                    100.0 * n / len(gold),
                )

    if chunk:
        tbl = pa.Table.from_pylist(chunk, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(out_part, schema=schema, compression="snappy")
        writer.write_table(tbl)
        chunk.clear()

    if writer is not None:
        writer.close()

    _save_ckpt(int(gold["gid"].max()) if not gold.empty else -1)
    if missing_uids:
        LOGGER.warning("%s rank %d: %d gids had missing users (filled with zeros).", out_base.name, rank, missing_uids)

    LOGGER.info("%s rank %d done: %s (elapsed %.1fs)", out_base.name, rank, out_part, time.time() - t0)


def merge_global_parts(
    *,
    paths: Paths,
    out_base: Path,
    world_size: int,
    cols_json: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = [out_base.with_name(f"{out_base.stem}_w{r}{out_base.suffix}") for r in range(world_size)]
    part_files = [p for p in parts if p.exists()]

    if not part_files:
        raise RuntimeError(f"No feature part files found to merge for base={out_base}")

    # Stream merge to avoid loading all shards into RAM at once.
    schema = pq.ParquetFile(part_files[0]).schema_arrow

    tmp_out = out_base.with_name(f"{out_base.name}.tmp")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except Exception:
            pass

    with pq.ParquetWriter(tmp_out, schema=schema, compression="snappy") as writer:
        for p in part_files:
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=256_000):
                tbl = pa.Table.from_batches([batch], schema=schema)
                writer.write_table(tbl)

    # Atomic replace
    os.replace(tmp_out, out_base)

    with open(cols_json, "w") as fh:
        json.dump(list(schema.names), fh)

    for r in range(world_size):
        try:
            parts[r].unlink(missing_ok=True)
            (paths.output_dir / f"feature_builder_ckpt_{out_base.stem}_{r}.json").unlink(missing_ok=True)
        except Exception:
            pass

    LOGGER.info("Merged %d shards → %s", world_size, out_base)



# ────────────────────────── CLI / ENTRY ────────────────────────── #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Global-static hypernet feature builder (leakage-safe, target-only)")
    ap.add_argument("--train_parquet", type=str, default=str(DEFAULT_DATA_DIR / "train_data_10000.parquet"))
    ap.add_argument("--val_parquet", type=str, default=str(DEFAULT_DATA_DIR / "val_data_10000.parquet"))
    ap.add_argument("--test_parquet", type=str, default=str(DEFAULT_DATA_DIR / "test_data_10000.parquet"))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--sent_model", type=str, default=str(DEFAULT_SENT_MODEL))
    ap.add_argument("--persona_model", type=str, default=str(DEFAULT_PERSONA_MODEL))

    ap.add_argument("--shards", type=int, default=1, help="Number of worker processes for replication stage")
    ap.add_argument("--sent_device", type=int, default=0, help="GPU id for sentiment model (use -1 for CPU)")
    ap.add_argument("--persona_device", type=int, default=0, help="GPU id for personality model (use -1 for CPU)")
    ap.add_argument("--force", action="store_true", help="Rebuild author table even if it exists")
    ap.add_argument("--resume", action="store_true", help="Resume replication stage from per-rank checkpoints")
    ap.add_argument("--disable_beast", action="store_true", help="Disable BEAST block (emit zeros for BEAST columns)")
    ap.add_argument(
        "--agg_split",
        type=str,
        default="train",
        choices=["train", "trainval", "all"],
        help="Which splits are allowed to contribute to author vectors (default=train for leakage-safety)",
    )

    ap.add_argument(
        "--split_outputs",
        type=str,
        default="all",
        choices=["none", "valtest", "all"],
        help=(
            "Which split-specific global feature tables to write in addition to the combined "
            "global_features_10000.parquet. "
            "none=combined only; valtest=combined+val+test; all=combined+train+val+test."
        ),
    )

    ap.add_argument(
        "--persona_activation",
        type=str,
        default="auto",
        choices=["auto", "sigmoid", "softmax", "regression"],
        help=(
            "How to convert personality logits to an author-level vector. "
            "'auto' uses the model config when available. "
            "'regression' uses the raw selected logits as the primary vector."
        ),
    )

    ap.add_argument(
        "--persona_idx_map",
        type=str,
        default="",
        help="Optional explicit comma-separated indices mapping model outputs to [A,O,C,E,N]. Example: '0,1,2,3,4'.",
    )

    ap.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    ap.add_argument("-f", "--f", help=argparse.SUPPRESS)
    return ap.parse_args()

def _replication_worker_entry(
    r: int,
    ws: int,
    resume_flag: bool,
    paths_dict: dict,
    out_base_str: str,
    split_paths_strs: List[str],
) -> None:
    if r != 0:
        logging.getLogger().setLevel(logging.WARNING)

    local_paths = Paths(
        train_parquet=Path(paths_dict["train_parquet"]),
        val_parquet=Path(paths_dict["val_parquet"]),
        test_parquet=Path(paths_dict["test_parquet"]),
        output_dir=Path(paths_dict["output_dir"]),
        sent_model=Path(paths_dict["sent_model"]),
        persona_model=Path(paths_dict["persona_model"]),
    )

    replicate_to_global_parts(
        rank=r,
        world_size=ws,
        paths=local_paths,
        out_base=Path(out_base_str),
        split_paths=[Path(p) for p in split_paths_strs],
        resume=resume_flag,
    )
    
def _run_replication_job(
    *,
    paths: Paths,
    out_base: Path,
    cols_json: Path,
    split_paths: Sequence[Path],
    world_size: int,
    resume: bool,
) -> None:
    if world_size <= 1:
        replicate_to_global_parts(rank=0, world_size=1, paths=paths, out_base=out_base, split_paths=split_paths, resume=resume)
        merge_global_parts(paths=paths, out_base=out_base, world_size=1, cols_json=cols_json)
        return

    import multiprocessing as mp

    LOGGER.info("Launching %d replication workers for %s …", world_size, out_base.name)
    ctx = mp.get_context("spawn")

    paths_dict = {
        "train_parquet": str(paths.train_parquet),
        "val_parquet": str(paths.val_parquet),
        "test_parquet": str(paths.test_parquet),
        "output_dir": str(paths.output_dir),
        "sent_model": str(paths.sent_model),
        "persona_model": str(paths.persona_model),
    }

    procs: List[mp.Process] = []
    for r in range(world_size):
        p = ctx.Process(
            target=_replication_worker_entry,
            args=(
                r,
                world_size,
                bool(resume),
                paths_dict,
                str(out_base),
                [str(pth) for pth in split_paths],
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Replication worker failed for {out_base.name} with exit code {p.exitcode}")

    merge_global_parts(paths=paths, out_base=out_base, world_size=world_size, cols_json=cols_json)

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))

    cpu_threads = int(os.getenv("CPU_THREADS", "8"))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
    LOGGER.info("Using %d CPU threads", cpu_threads)

    paths = Paths(
        train_parquet=Path(args.train_parquet),
        val_parquet=Path(args.val_parquet),
        test_parquet=Path(args.test_parquet),
        output_dir=Path(args.output_dir),
        sent_model=Path(args.sent_model),
        persona_model=Path(args.persona_model),
    )

    verify_split_disjointness(paths)

    build_author_table(
        paths,
        sent_device_id=int(args.sent_device),
        persona_device_id=int(args.persona_device),
        force=bool(args.force),
        disable_beast=bool(args.disable_beast),
        agg_split=str(args.agg_split),
        persona_activation=str(args.persona_activation),
        persona_idx_map=str(args.persona_idx_map),
    )

    world_size = int(args.shards)
    split_outputs = str(getattr(args, "split_outputs", "all")).strip().lower()

    jobs: List[Tuple[Path, Path, List[Path]]] = []

    # 1) ALWAYS emit combined global table across train+val+test gids
    jobs.append(
        (
            paths.global_features_parquet,
            paths.global_features_cols_json,
            [paths.train_parquet, paths.val_parquet, paths.test_parquet],
        )
    )

    # 2) Optionally emit split-specific tables (for leakage-safe training/eval wiring)
    if split_outputs in {"valtest", "all"}:
        jobs.append((paths.global_features_val_parquet, paths.global_features_val_cols_json, [paths.val_parquet]))
        jobs.append((paths.global_features_test_parquet, paths.global_features_test_cols_json, [paths.test_parquet]))

    if split_outputs == "all":
        jobs.append((paths.global_features_train_parquet, paths.global_features_train_cols_json, [paths.train_parquet]))

    for out_base, cols_json, split_paths in jobs:
        _run_replication_job(
            paths=paths,
            out_base=out_base,
            cols_json=cols_json,
            split_paths=split_paths,
            world_size=world_size,
            resume=bool(args.resume),
        )

    LOGGER.info("Completed build.")


if __name__ == "__main__":
    main()