# UNSW-NB15 feature schema

Each `train.csv` / `x_test.csv` has 42 feature columns. `train.csv` additionally
has a trailing binary `label` column (`1` = attack, `0` = normal); `x_test.csv`
has no label.

The original dataset ships 45 columns; two are removed here: `id` (a row index)
and `attack_cat` (the multiclass attack category, dropped because it leaks the
binary `label`). The binary target is exported as `label`.

## Categorical (string) features
- `proto` — transport/network protocol (e.g. `tcp`, `udp`, `arp`, `ospf`, …).
- `service` — application-layer service (e.g. `http`, `dns`, `ftp`, `smtp`,
  `ssl`, or `-` when none). **Test may contain unseen values.**
- `state` — connection state (e.g. `FIN`, `CON`, `INT`, `REQ`, `RST`, …).

## Numeric features

Flow / basic connection: `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`,
`sttl`, `dttl`, `sloss`, `dloss`.

Content / timing: `sload`, `dload`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`,
`stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`,
`trans_depth`, `response_body_len`.

Connection-count / behavioural (aggregates over recent connections):
`ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`,
`ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_src_ltm`, `ct_srv_dst`,
`ct_flw_http_mthd`, `ct_ftp_cmd`, `is_ftp_login`, `is_sm_ips_ports`.

## Notes
- `proto` / `service` / `state` are the only categorical columns; encode them so
  values unseen in training are handled (mapped to a reserved code), not dropped.
- Byte / load / duration features are heavy-tailed; consider log scaling for
  linear models. Tree models are scale-invariant.
- `is_ftp_login`, `is_sm_ips_ports` are 0/1 flags; `ct_*` are small counts.
