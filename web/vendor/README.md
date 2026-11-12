# Vendored frontend libraries

The lynx page must work without internet, so the libraries are stored here.
Do not load them from a CDN.

| Library | Version | Released | License | Source |
| --- | --- | --- | --- | --- |
| three.js | 0.186.1 (r186) | 2026-09-24 | MIT, see `three/LICENSE` | npm package `three@0.186.1` |

Files copied from the npm package, unchanged:

| File here | File in the package |
| --- | --- |
| `three/three.module.js`, `three/three.core.js` | `build/` |
| `three/addons/loaders/GLTFLoader.js` | `examples/jsm/loaders/` |
| `three/addons/utils/BufferGeometryUtils.js`, `three/addons/utils/SkeletonUtils.js` | `examples/jsm/utils/` (imported by `GLTFLoader.js`) |
| `three/addons/environments/RoomEnvironment.js` | `examples/jsm/environments/` |

To update: run `npm pack three@<version>`, copy the same files, and change this table.
Use a pinned version that is at least a few weeks old.
