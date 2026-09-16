# NSL-KDD feature schema

Each `train.csv` / `x_test.csv` has 41 feature columns (below) in this order.
`train.csv` additionally has a trailing binary `label` column (`1` = attack of
the instance's family, `0` = normal); `x_test.csv` has no label.

## Categorical (string) features
- `protocol_type` — `tcp`, `udp`, `icmp`
- `service` — network service on the destination (e.g. `http`, `ftp_data`,
  `private`, `domain_u`, …); ~70 values. **Test may contain unseen values.**
- `flag` — connection status (e.g. `SF`, `S0`, `REJ`, `RSTR`, …)

## Numeric features
Basic connection: `duration`, `src_bytes`, `dst_bytes`, `land`,
`wrong_fragment`, `urgent`.

Content: `hot`, `num_failed_logins`, `logged_in`, `num_compromised`,
`root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`,
`num_access_files`, `num_outbound_cmds`, `is_host_login`, `is_guest_login`.

Time-based traffic (2-second window): `count`, `srv_count`, `serror_rate`,
`srv_serror_rate`, `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`,
`diff_srv_rate`, `srv_diff_host_rate`.

Host-based traffic: `dst_host_count`, `dst_host_srv_count`,
`dst_host_same_srv_rate`, `dst_host_diff_srv_rate`,
`dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`,
`dst_host_serror_rate`, `dst_host_srv_serror_rate`, `dst_host_rerror_rate`,
`dst_host_srv_rerror_rate`.

## Notes
- `num_outbound_cmds` is constant (0) in NSL-KDD — safe to drop.
- Rate features are already in `[0, 1]`; byte/count features are heavy-tailed
  (consider log scaling).
- The `u2r` and `r2l` instances are highly imbalanced with few positives;
  class weighting or resampling matters there.
