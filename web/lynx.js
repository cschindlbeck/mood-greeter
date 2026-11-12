// The Christmas lynx: loads the model, drives a few bones and follows the events
// from the Python process (/events). No network access besides this server.
//
// Debug URL parameters (for tuning at a desk, for example /?open=0.8&still=1):
//   open=0..1    hold the mouth open by this amount
//   mood=<name>  hold a mood pose (happy, sad, angry, fear, surprise, disgust, neutral)
//   present=1    act as if a person is there
//   still=1      no idle movement (for screenshots)
//   noevents=1   do not connect to /events
//   jaw, lip, band, mx, mw   override the mouth constants below
// Keys: q = stop the Python process, [ and ] = shift the lip sync by 20 ms.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const params = new URLSearchParams(location.search);
const debug = {
  open: params.has('open') ? Number(params.get('open')) : null,
  mood: params.get('mood'),
  present: params.get('present') === '1',
  still: params.get('still') === '1',
  noEvents: params.get('noevents') === '1',
};

const MODEL_URL = '/assets/luchs-emerald-liegend.glb';
const FPS = 30; // Enough for a mascot. Leaves the CPU to the mood detection
const MAX_PIXEL_RATIO = 1.5;

// Framing: a close-up of the face. The model lies, so the body stays out of view.
const FACE_CENTER = new THREE.Vector3(-0.15, 0.83, 0.9);
const FACE_HALF_HEIGHT = 0.39; // Chin to hat tip, with a margin
const FACE_HALF_WIDTH = 0.36;
const CAMERA_OFFSET = new THREE.Vector3(0.12, 0.05, 2.6);

// Mouth. The jaw bone of the model also moves the nose and there is no mouth
// cavity. So at load time the skin weights above the lip line go to the head
// bone, and the shader darkens the lip band while it stretches open.
const JAW_MAX_ANGLE = Number(params.get('jaw') ?? 0.42); // Radians at full loudness
const LIP_LINE = Number(params.get('lip') ?? 0.012); // Height of the lip line in jaw bone space (metres)
const LIP_BAND = Number(params.get('band') ?? 0.004); // Half height of the band that stretches
const MOUTH_CENTER_X = Number(params.get('mx') ?? 0); // Mouth centre in jaw bone space
const MOUTH_HALF_WIDTH = Number(params.get('mw') ?? 0.09);
const MOUTH_COLOR = new THREE.Color(0x2a0508);
const LIP_LOOKAHEAD = 0.05; // Seconds. Makes up for the smoothing and one frame
const MOUTH_ATTACK = 0.03; // Seconds to open
const MOUTH_RELEASE = 0.07; // Seconds to close

// Head and ear pose per mood, in radians. Kept subtle on purpose.
// pitch: + nose up. yaw: turn to the side. roll: tilt to the side.
// ear: + forward, - back. splay: + ears apart and down.
const MOODS = {
  happy: { pitch: 0.05, ear: 0.2, splay: -0.05, nod: 0.06 },
  surprise: { pitch: 0.13, ear: 0.3, splay: -0.1 },
  sad: { pitch: -0.15, roll: 0.07, ear: -0.25, splay: 0.3 },
  angry: { pitch: -0.08, ear: -0.4, splay: 0.12, shake: 0.05 },
  fear: { pitch: -0.05, ear: -0.35, splay: 0.22, tremble: 0.012 },
  disgust: { pitch: 0.02, yaw: 0.15, roll: -0.09, ear: -0.15, splay: 0.15 },
  neutral: { roll: 0.09, ear: 0.1 },
};
const ATTENTIVE = { pitch: 0.04, ear: 0.15 }; // A person is there, mood not known yet
const ANALYSING = { pitch: 0.03, roll: 0.08, ear: 0.22 }; // Reading the mood: head tilted, ears up
const RELAXED = {}; // Nobody there

// ---------------------------------------------------------------- scene

const canvas = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'low-power' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
const scene = new THREE.Scene();
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.add(new THREE.HemisphereLight(0xffffff, 0x35505a, 0.9));
const keyLight = new THREE.DirectionalLight(0xfff1dd, 1.8);
keyLight.position.set(1.5, 2.5, 4);
scene.add(keyLight);
const camera = new THREE.PerspectiveCamera(20, 1, 0.1, 30);

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  // Fit the face box in both directions
  const half = Math.max(FACE_HALF_HEIGHT, FACE_HALF_WIDTH / camera.aspect);
  camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(half / CAMERA_OFFSET.length()));
  camera.position.copy(FACE_CENTER).add(CAMERA_OFFSET);
  camera.lookAt(FACE_CENTER);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// Snow falls behind the head only, so it never hides the face.
const SNOW_COUNT = 140;
const SNOW_BOX = { x: 3.2, yMin: -0.2, yMax: 2.4, zMin: -3.0, zMax: 0.2 };
const snowSpeed = new Float32Array(SNOW_COUNT);
const snow = makeSnow();
scene.add(snow);

function makeSnow() {
  const positions = new Float32Array(SNOW_COUNT * 3);
  for (let i = 0; i < SNOW_COUNT; i++) {
    positions[i * 3] = FACE_CENTER.x + (Math.random() - 0.5) * SNOW_BOX.x;
    positions[i * 3 + 1] = THREE.MathUtils.lerp(SNOW_BOX.yMin, SNOW_BOX.yMax, Math.random());
    positions[i * 3 + 2] = THREE.MathUtils.lerp(SNOW_BOX.zMin, SNOW_BOX.zMax, Math.random());
    snowSpeed[i] = 0.08 + Math.random() * 0.1;
  }
  const dot = document.createElement('canvas');
  dot.width = dot.height = 32;
  const context = dot.getContext('2d');
  const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, 32, 32);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    map: new THREE.CanvasTexture(dot), size: 0.07, transparent: true, opacity: 0.85, depthWrite: false,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  return points;
}

function updateSnow(time, dt) {
  const position = snow.geometry.attributes.position;
  for (let i = 0; i < SNOW_COUNT; i++) {
    let y = position.getY(i) - snowSpeed[i] * dt;
    if (y < SNOW_BOX.yMin) y = SNOW_BOX.yMax;
    position.setY(i, y);
    position.setX(i, position.getX(i) + Math.sin(time * 0.6 + i) * 0.02 * dt);
  }
  position.needsUpdate = true;
}

// ---------------------------------------------------------------- model

const bones = {}; // name -> { bone, base quaternion }
const mouthUniform = { value: 0 };
let modelReady = false;

new GLTFLoader().load(MODEL_URL, (gltf) => {
  const root = gltf.scene;
  let mesh = null;
  root.traverse((object) => {
    if (object.name === 'Liegeflaeche') object.visible = false; // The lying surface
    if (object.isSkinnedMesh) mesh = object;
  });
  mesh.frustumCulled = false; // The bounding box is the one of the standing bind pose
  applyFirstFrame(root, gltf.animations[0]);
  for (const name of ['head', 'neck01', 'jaw', 'ear_baseL', 'ear_baseR', 'ear_tipL', 'ear_tipR']) {
    const bone = root.getObjectByName(name);
    bones[name] = { bone, base: bone.quaternion.clone() };
  }
  prepareMouth(mesh);
  bones.head.bone.add(makeHat());
  scene.add(root);
  modelReady = true;
}, undefined, () => {
  // The model is not in git. `make setup` fetches it.
  document.getElementById('status').textContent = 'The lynx model is missing. Run "make setup".';
});

// The model stands in its rest pose. The first frame of the clip `Liegen_Idle`
// is the lying pose. The clip itself does not play: the movement is procedural.
function applyFirstFrame(root, clip) {
  for (const track of clip.tracks) {
    const split = track.name.lastIndexOf('.');
    const target = root.getObjectByName(track.name.slice(0, split));
    const property = target && target[track.name.slice(split + 1)];
    if (property && property.fromArray) property.fromArray(track.values);
  }
  root.updateMatrixWorld(true);
}

function prepareMouth(mesh) {
  const skeleton = mesh.skeleton;
  const jawIndex = skeleton.bones.indexOf(bones.jaw.bone);
  const headIndex = skeleton.bones.indexOf(bones.head.bone);
  const geometry = mesh.geometry;
  const position = geometry.attributes.position;
  const skinIndex = geometry.attributes.skinIndex;
  const skinWeight = geometry.attributes.skinWeight;
  const mouth = new Float32Array(position.count * 2); // x: 0 above the lips .. 1 below, y: 1 at the mouth
  const point = new THREE.Vector3();
  for (let i = 0; i < position.count; i++) {
    let jawSlot = -1;
    let headSlot = -1;
    let freeSlot = -1;
    for (let slot = 0; slot < 4; slot++) {
      const weight = skinWeight.getComponent(i, slot);
      if (weight === 0) freeSlot = slot;
      else if (skinIndex.getComponent(i, slot) === jawIndex) jawSlot = slot;
      else if (skinIndex.getComponent(i, slot) === headIndex) headSlot = slot;
    }
    if (jawSlot < 0) continue;
    // The vertex in the space of the jaw bone: z is up, y points to the nose
    point.fromBufferAttribute(position, i).applyMatrix4(mesh.bindMatrix).applyMatrix4(skeleton.boneInverses[jawIndex]);
    const lower = 1 - THREE.MathUtils.smoothstep(point.z, LIP_LINE - LIP_BAND, LIP_LINE + LIP_BAND);
    const side = Math.abs(point.x - MOUTH_CENTER_X);
    mouth[i * 2] = lower;
    mouth[i * 2 + 1] = 1 - THREE.MathUtils.smoothstep(side, MOUTH_HALF_WIDTH * 0.6, MOUTH_HALF_WIDTH);
    // Only the part below the lip line follows the jaw. The rest goes to the head.
    const weight = skinWeight.getComponent(i, jawSlot);
    if (headSlot < 0) headSlot = freeSlot;
    if (headSlot < 0) {
      if (lower < 0.5) skinIndex.setComponent(i, jawSlot, headIndex);
      continue;
    }
    skinIndex.setComponent(i, headSlot, headIndex);
    skinWeight.setComponent(i, headSlot, skinWeight.getComponent(i, headSlot) + weight * (1 - lower));
    skinWeight.setComponent(i, jawSlot, weight * lower);
  }
  skinIndex.needsUpdate = true;
  skinWeight.needsUpdate = true;
  geometry.setAttribute('aMouth', new THREE.BufferAttribute(mouth, 2));

  mesh.material.onBeforeCompile = (shader) => {
    shader.uniforms.uMouthOpen = mouthUniform;
    shader.uniforms.uMouthColor = { value: MOUTH_COLOR };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec2 aMouth;\nvarying vec2 vMouth;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvMouth = aMouth;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uMouthOpen;\nuniform vec3 uMouthColor;\nvarying vec2 vMouth;')
      .replace('#include <map_fragment>', `#include <map_fragment>
        float lipBand = smoothstep(0.03, 0.15, vMouth.x) * (1.0 - smoothstep(0.8, 0.95, vMouth.x));
        diffuseColor.rgb = mix(diffuseColor.rgb, uMouthColor, lipBand * vMouth.y * smoothstep(0.0, 0.35, uMouthOpen));`);
  };
  mesh.material.needsUpdate = true;
}

// A simple low-poly Santa hat. It sits between the ears and never covers the face.
function makeHat() {
  const red = new THREE.MeshStandardMaterial({ color: 0xc8141f, roughness: 0.85, flatShading: true });
  const white = new THREE.MeshStandardMaterial({ color: 0xf7f7f2, roughness: 0.95, flatShading: true });
  const hat = new THREE.Group();

  const height = 0.25;
  const cone = new THREE.ConeGeometry(0.105, height, 12, 6, true);
  cone.translate(0, height / 2, 0);
  const tip = new THREE.Vector3();
  const position = cone.attributes.position;
  for (let i = 0; i < position.count; i++) {
    // Bend the upper part to the side, so the tip hangs down
    const h = position.getY(i) / height;
    position.setX(i, position.getX(i) - 0.11 * h * h);
    position.setY(i, position.getY(i) - 0.07 * h * h * h);
    if (h > 0.99) tip.set(position.getX(i), position.getY(i), position.getZ(i));
  }
  cone.computeVertexNormals();
  hat.add(new THREE.Mesh(cone, red));

  const brim = new THREE.Mesh(new THREE.TorusGeometry(0.105, 0.03, 6, 14), white);
  brim.rotation.x = Math.PI / 2;
  hat.add(brim);

  const pompom = new THREE.Mesh(new THREE.IcosahedronGeometry(0.04, 1), white);
  pompom.position.copy(tip);
  hat.add(pompom);

  // Head bone space: z is up, y points to the nose. The hat is built with y up.
  hat.rotation.set(Math.PI / 2 - 0.15, 0, 0.12, 'ZYX');
  hat.position.set(0.016, 0.115, 0.215);
  return hat;
}

// ---------------------------------------------------------------- events

const caption = document.getElementById('caption');
const statusLabel = document.getElementById('status');
const cameraImage = document.getElementById('camera');

const show = { present: debug.present, phase: 'idle', mood: debug.mood, moodSince: 0 };
let speech = null; // { id, envelope, step, duration, startAt (ms on the performance clock) }
let lipOffset = Number(localStorage.getItem('lynx.lipOffsetMs') || 0) / 1000; // + moves the mouth later
let eventSource = null;
let lastEventAt = performance.now();
let stopped = false;

function handleEvent(event) {
  if (event.type === 'state') {
    if (!debug.present) show.present = event.present;
    show.phase = event.phase;
    if (!debug.mood && event.mood !== show.mood) {
      show.mood = event.mood;
      show.moodSince = performance.now();
    }
  } else if (event.type === 'speech_start') {
    speech = {
      id: event.id,
      envelope: event.envelope,
      step: event.step,
      duration: event.duration,
      startAt: performance.now() + event.delay * 1000,
    };
    caption.textContent = event.text;
    caption.classList.add('on');
  } else if (event.type === 'speech_end') {
    if (speech && speech.id === event.id) endSpeech();
  }
}

function endSpeech() {
  speech = null;
  caption.classList.remove('on');
}

function offline(message) {
  endSpeech();
  if (!debug.present) show.present = false;
  if (!debug.mood) show.mood = null;
  show.phase = 'idle';
  statusLabel.textContent = message;
}

function connect() {
  if (eventSource) eventSource.close();
  lastEventAt = performance.now();
  eventSource = new EventSource('/events');
  eventSource.onopen = () => {
    statusLabel.textContent = '';
    loadCamera();
  };
  eventSource.onmessage = (message) => {
    lastEventAt = performance.now();
    handleEvent(JSON.parse(message.data));
  };
  // EventSource connects again by itself. Until then the lynx sits idle.
  eventSource.onerror = () => offline(stopped ? 'The lynx has stopped.' : 'Waiting for the camera…');
}

function loadCamera() {
  cameraImage.src = '/camera.mjpg?' + Date.now();
}
cameraImage.onload = () => { cameraImage.style.visibility = 'visible'; };
cameraImage.onerror = () => {
  cameraImage.style.visibility = 'hidden';
  setTimeout(() => { if (!stopped) loadCamera(); }, 2000);
};

if (!debug.noEvents) {
  connect();
  // The server pings every 5 s. Silence means the connection hangs: start over.
  setInterval(() => {
    if (!stopped && performance.now() - lastEventAt > 15000) {
      offline('Waiting for the camera…');
      connect();
    }
  }, 5000);
}

window.addEventListener('keydown', (event) => {
  if (event.key === 'q') {
    stopped = true;
    fetch('/quit', { method: 'POST', headers: { 'X-Lynx': '1' } }).catch(() => {});
    offline('The lynx has stopped.');
  } else if (event.key === '[' || event.key === ']') {
    lipOffset += event.key === ']' ? 0.02 : -0.02;
    localStorage.setItem('lynx.lipOffsetMs', String(Math.round(lipOffset * 1000)));
    statusLabel.textContent = `Lip sync offset: ${Math.round(lipOffset * 1000)} ms`;
  }
});

// ---------------------------------------------------------------- animation

// How far the mouth should be open now, 0..1.
function mouthTarget(now) {
  if (debug.open !== null) return debug.open;
  if (!speech) return 0;
  const time = (now - speech.startAt) / 1000 - lipOffset + LIP_LOOKAHEAD;
  if (time > (speech.duration ?? 30) + 3) {
    endSpeech(); // The end event got lost
    return 0;
  }
  if (!speech.envelope) {
    // No envelope (plain `say`): syllable-like noise while the lynx speaks
    return Math.max(0, 0.45 + 0.35 * Math.sin(now * 0.031) + 0.25 * Math.sin(now * 0.0173 + 1.7));
  }
  const index = time / speech.step;
  if (index < 0 || index >= speech.envelope.length - 1) return 0;
  const low = Math.floor(index);
  return THREE.MathUtils.lerp(speech.envelope[low], speech.envelope[low + 1], index - low) / 100;
}

const pose = { pitch: 0, yaw: 0, roll: 0, ear: 0, splay: 0 };
const flick = { ear: 'ear_baseL', start: -10, next: 3 };
let mouthOpen = 0;
const offset = new THREE.Quaternion();
const euler = new THREE.Euler();

function setBone(name, x, y, z) {
  const entry = bones[name];
  entry.bone.quaternion.copy(entry.base).multiply(offset.setFromEuler(euler.set(x, y, z, 'ZXY')));
}

function animate(time, dt, now) {
  // 1. Mouth
  const target = mouthTarget(now);
  const tau = target > mouthOpen ? MOUTH_ATTACK : MOUTH_RELEASE;
  mouthOpen += (target - mouthOpen) * (1 - Math.exp(-dt / tau));
  if (mouthOpen < 0.003) mouthOpen = 0;
  mouthUniform.value = mouthOpen;
  setBone('jaw', -mouthOpen * JAW_MAX_ANGLE, 0, 0);

  // 2. Pose of the mood, approached softly
  const mood = MOODS[show.mood] || (show.phase === 'analysing' ? ANALYSING : show.present ? ATTENTIVE : RELAXED);
  const blend = 1 - Math.exp(-dt * 4);
  for (const key of Object.keys(pose)) pose[key] += ((mood[key] || 0) - pose[key]) * blend;

  // 3. Idle life: slow look-around, smaller when the lynx looks at a person
  const wander = debug.still ? 0 : (show.present ? 0.35 : 1);
  let pitch = pose.pitch + wander * 0.035 * (Math.sin(time * 0.43) + 0.5 * Math.sin(time * 1.07 + 2));
  let yaw = pose.yaw + wander * 0.09 * (Math.sin(time * 0.29) + 0.5 * Math.sin(time * 0.71 + 1.3));
  const roll = pose.roll + wander * 0.03 * Math.sin(time * 0.37 + 0.6);

  // 4. Mood gestures for the first seconds, and a small bob with the voice
  const moodAge = (now - show.moodSince) / 1000;
  const gesture = show.mood && !debug.still ? Math.exp(-moodAge / 1.8) : 0;
  pitch += gesture * (mood.nod || 0) * Math.sin(moodAge * 9);
  yaw += gesture * (mood.shake || 0) * Math.sin(moodAge * 11);
  yaw += (mood.tremble || 0) * Math.sin(time * 38) * (debug.still ? 0 : 1);
  pitch -= mouthOpen * 0.02;

  setBone('head', pitch, roll, yaw);
  setBone('neck01', debug.still ? 0 : 0.012 * Math.sin(time * 1.4), 0, 0); // Breathing

  // 5. Ears: mood pose plus a quick flick of one ear now and then
  if (!debug.still && time > flick.next) {
    flick.ear = Math.random() < 0.5 ? 'ear_baseL' : 'ear_baseR';
    flick.start = time;
    flick.next = time + 3.5 + Math.random() * 6;
  }
  const flickPhase = (time - flick.start) / 0.28;
  const flickAngle = flickPhase >= 0 && flickPhase <= 1 ? -0.3 * Math.sin(Math.PI * flickPhase) : 0;
  setBone('ear_baseL', pose.ear + (flick.ear === 'ear_baseL' ? flickAngle : 0), 0, -pose.splay);
  setBone('ear_baseR', pose.ear + (flick.ear === 'ear_baseR' ? flickAngle : 0), 0, pose.splay);
  setBone('ear_tipL', pose.ear * 0.5, 0, 0);
  setBone('ear_tipR', pose.ear * 0.5, 0, 0);
}

let lastFrame = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  if (now - lastFrame < 1000 / FPS - 2) return;
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  const time = now / 1000;
  if (modelReady) animate(time, dt, now);
  updateSnow(time, dt);
  renderer.render(scene, camera);
}
requestAnimationFrame(frame);

// A lost WebGL context comes back by itself in most cases. If not, start over.
canvas.addEventListener('webglcontextlost', (event) => {
  event.preventDefault();
  setTimeout(() => { if (renderer.getContext().isContextLost()) location.reload(); }, 5000);
});

// For automated checks and screenshots
window.lynxReady = () => modelReady;
window.lynxDebug = () => ({
  mouth: Number(mouthOpen.toFixed(2)), speaking: speech !== null, present: show.present, mood: show.mood,
  phase: show.phase, status: statusLabel.textContent, camera: cameraImage.style.visibility === 'visible', caption: caption.className,
});
