# Assets

`luchs-emerald-liegend.glb` is the LYNQTECH mascot (emerald lynx, lying). It is company property.
This repo is public, so the file is not in git (see `.gitignore`). `make setup` fetches it with `gh`
from the private repo (`MODEL_REPO`, `MODEL_REF`, `MODEL_PATH` in the `Makefile`). You need `gh auth login`
with access to that repo, and internet once. Or copy the file here by hand.

Do not commit the model or the Blender sources.

What the page uses from it (see `web/lynx.js`):

- Mesh `Mesh0` (the lynx). Mesh `Liegeflaeche` (the lying surface) is hidden.
- The first frame of the clip `Liegen_Idle` as the lying pose. The rest pose is a standing lynx.
- Bones `head`, `neck01`, `jaw`, `ear_base.L/R`, `ear_tip.L/R`. three.js removes the dots from the names (`ear_baseL`).
- There are no morph targets and there is no mouth cavity. The `jaw` bone also moves the nose.
  `web/lynx.js` corrects this at load time.
