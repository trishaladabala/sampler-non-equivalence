#!/usr/bin/env python3
import re

with open('main.log') as f:
    log = f.read()

errors = re.findall(r'^! .*', log, re.M)
undef_refs = re.findall(r'Reference [^\n]+ undefined', log)
undef_cits = re.findall(r'Citation [^\n]+ undefined', log)
overfulls = re.findall(r'Overfull \\hbox [^\n]+', log)

print(f"Errors count: {len(errors)}")
for e in errors:
    print("  ERROR:", e)

print(f"\nUndefined References count: {len(undef_refs)}")
for r in undef_refs:
    print("  REF:", r)

print(f"\nUndefined Citations count: {len(undef_cits)}")
for c in undef_cits:
    print("  CIT:", c)

print(f"\nOverfull boxes count: {len(overfulls)}")
for o in overfulls:
    print("  OVERFULL:", o)
