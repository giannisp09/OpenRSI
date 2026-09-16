# Phishing-URL feature schema

Each `train.csv` / `x_test.csv` has 111 numeric feature columns. `train.csv`
additionally has a trailing binary `label` column (`1` = phishing, `0` =
legitimate); `x_test.csv` has no label. The dataset is roughly balanced (~52%
phishing).

All features are numeric. A value of **`-1` means "could not be resolved"**
(a DNS / WHOIS / traffic lookup that failed or was unavailable), not a real
magnitude — handle it as missing/unknown rather than as a number.

## Character-count features (per URL component)
For each component — the whole URL, and separately its `domain`, `directory`,
`file`, and `params` — a count of each special character:
`qty_dot_*`, `qty_hyphen_*`, `qty_underline_*`, `qty_slash_*`,
`qty_questionmark_*`, `qty_equal_*`, `qty_at_*`, `qty_and_*`,
`qty_exclamation_*`, `qty_space_*`, `qty_tilde_*`, `qty_comma_*`,
`qty_plus_*`, `qty_asterisk_*`, `qty_hashtag_*`, `qty_dollar_*`,
`qty_percent_*` (suffix `_url`, `_domain`, `_directory`, `_file`, `_params`).

## Length / structure features
`length_url`, `domain_length`, `directory_length`, `file_length`,
`params_length`, `qty_tld_url`, `qty_vowels_domain`, `qty_params`,
`tld_present_params`, `qty_redirects`.

## Boolean / flag features (0 or 1)
`domain_in_ip` (host is a raw IP), `server_client_domain`,
`email_in_url`, `url_shortened`, `tls_ssl_certificate`, `domain_spf`.

## Host / DNS / traffic features
`time_response`, `qty_ip_resolved`, `qty_nameservers`, `qty_mx_servers`,
`ttl_hostname`, `time_domain_activation`, `time_domain_expiration`,
`asn_ip`, `qty_redirects`, `url_google_index`, `domain_google_index`.
These are the columns most likely to carry `-1` (lookup failed).

## Notes
- Byte/time/ASN features are heavy-tailed; consider log scaling for linear
  models. Tree models are scale-invariant.
- Because `-1` is an "unknown" sentinel across many features, an explicit
  missing-indicator (or replacing `-1` with NaN before imputation) often helps.
- The label column is named `phishing` in the raw source; it is exported here as
  `label` in `train.csv`.
