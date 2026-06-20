# debug/ — manual smoke scripts (git-ignored)

One tiny runnable script per build block, to *see* the code work by hand —
launch instances, drive an env, watch the loop. This is distinct from the
automated tests in `tests/` (which gate CI): these are for eyeballing behaviour.
Not shipped; not pushed.

Run, e.g.:

    python debug/block3_http_roundtrip.py

Each script is delivered alongside its block, and exercises that block's feature
end-to-end with the smallest possible code.
