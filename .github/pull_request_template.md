## Summary

<!-- What does this PR change and why? -->

## Changes

<!-- Bullet the key changes. -->

## How to test

```bash
make install
make pipeline        # end-to-end: data -> train -> detect -> optimize
make test            # pytest suite
```

## Checklist

- [ ] Tests pass (`make test`)
- [ ] Drift detection / optimization behaviour documented if changed
- [ ] Config defaults reviewed (`config/config.yaml`)
