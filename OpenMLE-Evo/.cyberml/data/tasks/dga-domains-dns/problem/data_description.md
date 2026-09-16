# DGA domain data schema

`train.csv` has two columns:

- `domain` — the domain name as a string, including its TLD
  (e.g. `mortiscontrastatim.com`, `google.com`, `kw1c3lrhs8.net`).
- `label` — binary target: `1` = DGA (malware-generated), `0` = legit (benign).

`x_test.csv` has only the `domain` column. The set is balanced (~50% DGA).

There are **no numeric feature columns** — you derive features from the string.

## Where the classes come from
- **legit (0):** real, popular domains (Alexa top-sites style).
- **DGA (1):** domains synthesized by 25+ malware families' generation
  algorithms (Conficker, Cryptolocker, Emotet, Gozi, Necurs, Ramnit, Tinba,
  Suppobox, …). Different families have different styles — some look purely
  random, some concatenate dictionary words — so a single simple threshold is
  not enough.

## Feature ideas (the task is engineering these)
- **Lengths & counts:** total length, label length (before the TLD), number of
  labels/dots, digit count and ratio, hyphen count.
- **Randomness:** Shannon entropy over characters; ratio of distinct characters;
  longest consonant run; vowel/consonant ratio.
- **n-grams:** character bigram/trigram frequency profile; fraction of bigrams
  that are rare/never seen in benign text; a `CountVectorizer`/`TfidfVectorizer`
  with `analyzer="char"` or `analyzer="char_wb"` and `ngram_range=(2,4)` over the
  domain label is a strong, compact representation.
- **TLD:** the TLD (`.com`, `.ru`, `.info`, …) can carry signal — treat it as a
  category separate from the label string.

## Notes
- Strip or separate the TLD before computing "randomness" features, so `.com`
  doesn't dominate short domains.
- Some DGA families use dictionary words (low entropy) — entropy alone will miss
  them; n-gram/language-model features catch them.
- Everything must be derived from the provided strings and the standard library /
  installed packages; do not fetch external corpora or models.
