"""The clips of a skinned cloud as one stage with a variant set.

USD's own answer to "pick one of these" is a variant set, so seventy-one
animations of the same bird are one prim with one set on it: whoever opens the
stage sees a `clip` set with seventy-one names in it and picks one. Nothing in
the engine or the viewer knows what a clip is -- `athenea view` offers whatever
variant sets a stage carries.

What makes it cheap is composition. The cloud (four million gaussians, 365 MB)
comes in once as a payload; each variant is a reference to that clip's 3 MB of
`skinningXforms`, and USD opens the one that is selected.

Two strengths had to be right, and they are why it is a payload and a
reference rather than sublayers:

  * a sublayer's opinion is local to the root layer stack, which beats any
    variant, so the base cloud may not be sublayered here -- the rest pose
    would win over the clip's samples;
  * a reference inside a variant is stronger than a payload, so the clip's
    time samples do win over the payloaded cloud's default value.

    python3 scripts/sparrow-clip-variants.py [assets] [out]

`assets` defaults to ~/tools/assets/Sparrow and `out` to
<assets>/SparrowClips.usda. Models do not go in this repository; this writes
beside them.
"""
import os
import re
import sys

assets = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/tools/assets/Sparrow')
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(assets, 'SparrowClips.usda')
cloud = 'Sparrow_vis.usdc'      # the baked visibility field's cloud: it relights
clips = sorted(f[:-len('_rig.usda')] for f in os.listdir(os.path.join(assets, 'clips'))
               if f.endswith('_rig.usda'))
if not clips:
    sys.exit('athenea: no clips/<name>_rig.usda under %s' % assets)


def declared(path):
    """A layer's startTimeCode and endTimeCode, from its header."""
    with open(path) as f:
        head = f.read(4096)
    first = re.search(r'startTimeCode = ([-0-9.]+)', head)
    last = re.search(r'endTimeCode = ([-0-9.]+)', head)
    return (float(first.group(1)) if first else 1.0, float(last.group(1)) if last else 1.0)


ranges = {c: declared(os.path.join(assets, 'clips', '%s_rig.usda' % c)) for c in clips}
first = min(r[0] for r in ranges.values())
last = max(r[1] for r in ranges.values())

lines = ['#usda 1.0', '(', '    defaultPrim = "World"', '    upAxis = "Z"',
         '    startTimeCode = %g' % first, '    endTimeCode = %g' % last,
         '    timeCodesPerSecond = 30', ')', '',
         '# The bird, once, with every clip it knows as a variant. The declared',
         '# range above is the longest clip; what a host should show is the range',
         '# the selected clip\'s samples occupy (StageRenderer::animationRange).',
         'def Xform "World" (',
         '    prepend payload = @./%s@</World>' % cloud,
         '    variants = {',
         '        string clip = "%s"' % clips[0],
         '    }',
         '    prepend variantSets = "clip"',
         ')', '{',
         '    def Scope "Lights"', '    {',
         '        def DomeLight "Sky"', '        {',
         '            float inputs:intensity = 1.15',
         '            color3f inputs:color = (0.72, 0.79, 0.95)', '        }',
         '        def DistantLight "Sun"', '        {',
         '            float inputs:intensity = 4.5',
         '            float inputs:angle = 1.2',
         '            color3f inputs:color = (1, 0.94, 0.82)',
         '            double3 xformOp:rotateXYZ = (-52, 0, 36)',
         '            uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]', '        }',
         '    }', '',
         '    variantSet "clip" = {']
for clip in clips:
    lines += ['        "%s" {' % clip,
              '            over "Splats" (',
              '                prepend references = @./clips/%s_rig.usda@</World/Splats>' % clip,
              '            )', '            {', '            }', '        }']
lines += ['    }', '}', '']
open(out, 'w').write('\n'.join(lines))
print('athenea: %d clips, %g..%g -> %s' % (len(clips), first, last, out))
