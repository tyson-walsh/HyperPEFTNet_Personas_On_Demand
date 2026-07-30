# Released data artifacts

Three artifacts back the experiments in *Personas on Demand: Hypernetwork Synthesis of Unseen
Users*: the per-author profile vectors, the thread-level split, and the cohort and axis labels.
All identifiers are integers. There is no bridge from any identifier back to a Reddit account name,
and no raw post text is included. The forum corpus itself is not redistributed; it is rebuilt from
the public Pushshift Reddit archive using the thread identifiers in `thread_split_10000.parquet`
and the preprocessing in `data_scripts/`.

## `profile_vectors_10000.parquet` (and `.csv`)

One row per author, 10,000 authors. Columns: `target_user_id` (integer) plus the 18 model-input
profile features. Each feature is standardized (z-scored) across the 10,000 authors, so the values
are what the hypernetwork consumes directly; no normalization side-car is needed. This is the input
vector `g_u` from the paper.

| feature | meaning |
| --- | --- |
| `gstat_user_len_mean` | mean reply length |
| `gstat_user_sr_max_share` | share of activity in the author's single most-used subreddit |
| `gstat_question_ratio` | fraction of sentences that are questions |
| `gstat_caps_ratio` | capitalization ratio |
| `gstat_readability_fk` | Flesch-Kincaid grade level |
| `gstat_link_ratio` | fraction of posts containing a link |
| `gstat_negation_ratio` | negation-word ratio |
| `gstat_subjectivity_ratio` | subjective-language ratio |
| `gstat_emoticon_ratio` | emoticon ratio |
| `gstat_contraction_ratio` | contraction ratio |
| `gstat_avg_word_len` | average word length |
| `gstat_long_word_ratio` | long-word ratio |
| `gstat_stopword_ratio` | stopword ratio |
| `gstat_rep_bigram_ratio` | repeated-bigram ratio |
| `gstat_hour_entropy` | entropy of the posting-hour distribution |
| `gstat_nocturnal_ratio` | fraction of activity during night hours |
| `gstat_circadian_mean` | circadian mean posting time |
| `gstat_reply_delay_std` | variability of reply delay |

The parquet and CSV hold the same table; the parquet preserves float32 dtypes.

## `thread_split_10000.parquet`

One row per thread, 2,997,033 threads. Columns: `gid` (integer thread identifier),
`target_user_id` (integer, the thread's target author), and `split` (`train`, `val`, or `test`).
The partition is disjoint by `gid`:

| split | threads |
| --- | --- |
| train | 2,097,117 |
| val | 599,923 |
| test | 299,993 |

To rebuild the corpus, pull each `gid`'s thread from the Pushshift archive and apply the
preprocessing in `data_scripts/hypernetwork_feature_builder_10000.py`.

## `labels/`

Cohort and axis labels, one file per behavioral axis. Every CSV has two columns,
`target_user_id` and `label`, where `label` is a short category name (for example `rage`, `empath`,
`polite`, `vulgar`). The axes cover sentiment (`labels_sentiment*`), politeness, self-focus, tempo,
curiosity, expressiveness, anxiety, hostility, and warmth; `xtreme_sent_labels_rage_empath_200.csv`
is the 200-author rage and empath extreme set. `labels_manifest.json` documents each family's
construction (score column, tail method, quantile boundaries, and counts); the `*_metadata.json`
files record the sentiment scorers.

## Provenance and privacy

Profile vectors are computed from the public Pushshift Reddit archive (2010 to 2016). Usernames are
irreversibly hashed with SHA-256 before any processing, and the hash is discarded before these files
are written, so the integer `target_user_id` is the only identifier present. No raw post text,
account names, or person-tied weights are released. See the paper's Data and Ethical Considerations
sections.
