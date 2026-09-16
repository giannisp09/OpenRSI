# TUANDROMD feature schema

Each `train.csv` / `x_test.csv` has 241 binary feature columns. `train.csv`
additionally has a trailing binary `label` column (`1` = malware, `0` =
goodware); `x_test.csv` has no label. The dataset is imbalanced toward malware
(~80% of rows).

All 241 features are binary (`0`/`1`) presence flags — there are no continuous
or categorical-string columns. Each column is one of:

## Requested Android permissions
Manifest permissions the app declares, one column per permission, e.g.
`SEND_SMS`, `RECEIVE_SMS`, `READ_SMS`, `ACCESS_FINE_LOCATION`,
`ACCESS_COARSE_LOCATION`, `READ_CONTACTS`, `READ_PHONE_STATE`,
`INSTALL_PACKAGES`, `ACCESS_ALL_DOWNLOADS`, `ACCESS_CACHE_FILESYSTEM`,
`ACCESS_CHECKIN_PROPERTIES`. A `1` means the app requests it.

## Decompiled API-call signatures
Whether the decompiled app invokes a given API, one column per signature, in JVM
descriptor form, e.g. `Landroid/telephony/SmsManager;->sendTextMessage` or
`Lorg/apache/http/impl/client/DefaultHttpClient;->execute`. A `1` means the call
appears in the app's bytecode.

## Notes
- Features are already `0`/`1`, so feature scaling is largely a no-op for tree
  models; it only affects linear models' regularization geometry.
- The space is **high-dimensional and sparse** — most flags are `0` for any
  given app. L1 regularization, mutual-information / chi-squared feature
  selection, or tree models with built-in feature importance tend to help.
- The classes are **imbalanced (~80% malware)** — consider `class_weight` and
  threshold tuning; F1 on the malware class (pos_label=1) is what is scored.
- One source row had an empty label and is dropped upstream; the `Label` column
  (`malware`/`goodware`) is mapped to `1`/`0` and exported as `label`.
