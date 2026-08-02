# Data layout

The final recipe needs two organizer archives:

```text
data/raw/BuzzSet_challenge.zip
data/raw/BuzzSet_challenge_testphase.tar
```

Verified identities:

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `BuzzSet_challenge.zip` | 35,261,367,058 | `964ad8ef28c52cf9e43d77ff08ad8dbc45eb56301745801b4e641ee93d3a4453` |
| `BuzzSet_challenge_testphase.tar` | 19,557,529,600 | `24cf58858a33b7f1d06af37cb06d155ac5a5becff1e452212b1b8740f031e846` |

The first archive supplies the train and validation images. The second archive
supplies the corrected train/validation annotations and FinalTest data. The
extended test-phase archive was distributed by the challenge organizer; this
repository does not mirror it.

The organizer-provided COCO files declare `CC BY-NC 4.0` in their `licenses`
field. See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) before
redistributing data or derived artifacts.

`bash scripts/prepare_data.sh` creates the layout expected by the configs:

```text
data/extracted/BuzzSet_challenge/{train,valid}/
data/extracted/testphase/BuzzSpot_testphase/annotations/
data/extracted/testphase/BuzzSpot_testphase/test_testphase/
```

The preparation script writes completion markers only after extraction and
required-path checks succeed. Removing a marker intentionally causes the
corresponding archive to be verified and extracted again.
