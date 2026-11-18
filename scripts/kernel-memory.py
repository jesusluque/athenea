#!/usr/bin/env python3
# Copyright (c) 2026 jesus luque.
"""What each generated kernel keeps in a thread's memory, with no device.

A kernel that keeps more in a thread's memory than the device gives a thread
is built by Metal without a word, and its threads read back what others wrote
(docs/decisions.md, 18 Nov, "What a thread's memory bears"). The device says
nothing of the frame it lays out, but Apple's front end does: the LLVM IR it
writes for a kernel's Metal source holds every local that stays memory as an
`alloca`, function by function. This reads them.

    ATHENEA_SHADER_DUMP=/tmp/dump athenea stage scene.usda ...     # the generated modules
    scripts/kernel-memory.py /tmp/dump [--shaders build/<preset>/shaders] [--filter path] [-v]

For every compute entry of every module in the dump: slangc to Metal, `xcrun
metal -S -emit-llvm -O2` to IR, and three sums in bytes --

    own     the entry's own frame (the kernel's static variables live there)
    chain   the deepest chain of calls, each function's allocas added
    all     every function the entry reaches, added

`all` is what the device's compiler has to find room for once it has inlined
everything, and the figure to hold down; `athenea_gpu_tests "[threadmemory]"`
measures what an array alone bears on the device at hand. `-v` lists the
largest frames and what is in them, which is where to look for what to take
out. Needs the Metal toolchain (xcrun metal) and slangc (~/tools/slang/bin or
$SLANGC).
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile


def split_fields(body):
    out, depth, cur = [], 0, ''
    for ch in body:
        if ch in '{[<(':
            depth += 1
        if ch in '}]>)':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Module:
    """The allocas and calls of one .ll file."""

    def __init__(self, text):
        self.types = {m.group(1): m.group(2).strip() for m in re.finditer(r'^(%[\w.":]+) = type (.*)$', text, re.M)}
        self.memo = {}
        self.frame, self.calls, self.allocas = {}, {}, {}
        cur = None
        for line in text.split('\n'):
            m = re.match(r'^define .*?@([\w.$"]+)\(', line)
            if m:
                cur = m.group(1)
                self.frame[cur], self.calls[cur], self.allocas[cur] = 0, set(), []
                continue
            if line.startswith('}'):
                cur = None
                continue
            if cur is None:
                continue
            m = re.search(r'= alloca (.*?)(?:, align \d+)?(?:, !.*)?$', line)
            if m:
                kind, count = m.group(1), 1
                k = re.match(r'(.*), (i32|i64) (\d+)$', kind)
                if k:
                    kind, count = k.group(1), int(k.group(3))
                size = self.size(kind)[0] * count
                self.frame[cur] += size
                self.allocas[cur].append((size, kind))
            for c in re.finditer(r'call [^@]*@([\w.$"]+)\(', line):
                self.calls[cur].add(c.group(1))

    def size(self, t):
        """(bytes, alignment) of an LLVM type as Metal lays it out."""
        t = t.strip()
        if t in self.memo:
            return self.memo[t]
        if t.endswith('*') or t == 'ptr':
            r = (8, 8)
        elif t in ('i1', 'i8'):
            r = (1, 1)
        elif t in ('i16', 'half'):
            r = (2, 2)
        elif t in ('i32', 'float'):
            r = (4, 4)
        elif t in ('i64', 'double'):
            r = (8, 8)
        elif t.startswith('%'):
            r = self.size(self.types[t]) if self.types.get(t, 'opaque') != 'opaque' else (8, 8)
        elif t.startswith('<{'):
            r = (sum(self.size(x)[0] for x in split_fields(t[2:-2])), 1)
        elif t.startswith('<'):
            n, e = re.match(r'<(\d+) x (.*)>$', t).groups()
            s, p = self.size(e)[0] * int(n), 1
            while p < s:
                p *= 2
            r = (p, min(p, 16))   # a float3 is sixteen bytes
        elif t.startswith('['):
            n, e = re.match(r'\[(\d+) x (.*)\]$', t).groups()
            s, a = self.size(e)
            r = (s * int(n), a)
        elif t.startswith('{'):
            off, al = 0, 1
            for x in split_fields(t[1:-1]):
                s, a = self.size(x)
                off = (off + a - 1) // a * a + s
                al = max(al, a)
            r = ((off + al - 1) // al * al, al)
        else:
            r = (8, 8)
        self.memo[t] = r
        return r

    def deepest(self, f, seen=(), memo=None):
        memo = {} if memo is None else memo
        if f in memo:
            return memo[f]
        if f in seen or f not in self.frame:
            return 0, []
        best = (0, [])
        for c in self.calls[f]:
            d = self.deepest(c, seen + (f,), memo)
            if d[0] > best[0]:
                best = d
        memo[f] = (self.frame[f] + best[0], [f] + best[1])
        return memo[f]

    def reached(self, f):
        got, todo = set(), [f]
        while todo:
            g = todo.pop()
            if g in got or g not in self.frame:
                continue
            got.add(g)
            todo += list(self.calls[g])
        return got


def short(name):
    """A mangled name's own part."""
    m = re.match(r'_Z\d+([A-Za-z_]\w*?)_\d+[A-Z]', name)
    return m.group(1) if m else name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump', help='a directory ATHENEA_SHADER_DUMP wrote')
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--shaders', default=os.path.join(here, '..', 'shaders'), help='the shader directory (a build\'s, where a module includes a dependency\'s header)')
    ap.add_argument('--filter', default='', help='only modules whose name holds this')
    ap.add_argument('--keep', default='', help='keep the Metal and the IR in this directory')
    ap.add_argument('-v', action='store_true', help='the largest frames of each kernel and what they hold')
    args = ap.parse_args()
    slangc = os.environ.get('SLANGC', os.path.expanduser('~/tools/slang/bin/slangc'))
    work = args.keep or tempfile.mkdtemp(prefix='kernel-memory-')
    os.makedirs(work, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(args.dump)):
        if not name.endswith('.slang') or args.filter not in name:
            continue
        path = os.path.join(args.dump, name)
        entries = re.findall(r'\[shader\("compute"\)\]\s*(?:\[numthreads\([^)]*\)\]\s*)?void\s+(\w+)', open(path).read())
        for entry in entries:
            base = os.path.join(work, name[:-6] + '_' + entry)
            made = subprocess.run([slangc, path, '-I', args.shaders, '-I', args.dump, '-target', 'metal', '-entry', entry,
                                   '-stage', 'compute', '-o', base + '.metal'], capture_output=True, text=True)
            if made.returncode != 0:
                errors = [l for l in made.stderr.split('\n') if l.startswith('error') or l.startswith('fatal')]
                print('%s %s: slangc failed: %s' % (name, entry, '; '.join(errors[:2])), file=sys.stderr)
                continue
            made = subprocess.run(['xcrun', 'metal', '-std=metal4.0', '-w', '-S', '-emit-llvm', '-O2', base + '.metal', '-o',
                                   base + '.ll'], capture_output=True, text=True)
            if made.returncode != 0:
                print('%s %s: xcrun metal failed: %s' % (name, entry, made.stderr.strip().split('\n')[-1]), file=sys.stderr)
                continue
            module = Module(open(base + '.ll').read())
            if entry not in module.frame:
                continue
            chain = module.deepest(entry)
            reach = module.reached(entry)
            total = sum(module.frame[f] for f in reach)
            rows.append((re.sub(r'^athenea_materials_[0-9a-f]+_?', '', name[:-6]) or name[:-6], entry, module.frame[entry], chain[0], total))
            if args.v:
                print('%s %s: own %d, chain %d, all %d' % (rows[-1][0], entry, module.frame[entry], chain[0], total))
                for f in sorted(reach, key=lambda f: -module.frame[f])[:12]:
                    if module.frame[f] == 0:
                        break
                    held = sorted(module.allocas[f], reverse=True)[:6]
                    print('  %6d %s: %s' % (module.frame[f], short(f), ', '.join('%d %s' % (s, t[:36]) for s, t in held)))
    width = max([len(r[0]) + len(r[1]) for r in rows] + [10]) + 2
    print('%-*s %8s %8s %8s' % (width, 'kernel', 'own', 'chain', 'all'))
    for module, entry, own, chain, total in rows:
        print('%-*s %8d %8d %8d' % (width, module + ' ' + entry, own, chain, total))


if __name__ == '__main__':
    main()
