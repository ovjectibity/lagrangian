<script lang="ts">
  import * as three from 'three';
  import { onMount } from 'svelte';
  import type { WorldBody, Body, Geom } from '../commondef/model';
  import type { ModelForward } from "../commondef/messages";
  import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

  const {
    width = 500, 
    height = 300,
    worldBody = undefined,
    simulUpdateData = undefined
  } = $props<{ width: number; height: number, worldBody?: WorldBody, simulUpdateData?: ModelForward}>();

  let canvas: HTMLCanvasElement;
  let scene: three.Scene;
  let camera: three.PerspectiveCamera;
  let cameraController: OrbitControls;
  let renderer: three.WebGLRenderer;
  let viewportEl: HTMLDivElement;
  let bodyObjectMap: Map<number, three.Object3D> = new Map();

  const resizeRenderer = () => {
    if (!renderer || !camera || !viewportEl) return;
    const nextWidth = viewportEl.clientWidth || width;
    const nextHeight = height;
    renderer.setSize(nextWidth, nextHeight, false);
    camera.aspect = nextWidth / nextHeight;
    camera.updateProjectionMatrix();
  };

  const renderBodyAbsolute = (body: Body, parent: three.Object3D) => {
    const group = new three.Group();
    if (body.name) {
      group.name = body.name;
    }
    if (body.body_id) {
        bodyObjectMap.set(body.body_id, group);
    }

    const pos = new three.Vector3();
    if (body.pos) {
      pos.set(body.pos[0], body.pos[1], body.pos[2]);
    }
    const quat = new three.Quaternion();
    if (body.quat) {
      quat.set(body.quat[1], body.quat[2], body.quat[3], body.quat[0]);
    } else if (body.euler) {
      const euler = new three.Euler(
        three.MathUtils.degToRad(body.euler[0]),
        three.MathUtils.degToRad(body.euler[1]),
        three.MathUtils.degToRad(body.euler[2])
      );
      quat.setFromEuler(euler);
    }
    
    const bodyWorldMatrix = new three.Matrix4().compose(pos, quat, new three.Vector3(1, 1, 1));
    const parentWorldMatrixInverse = new three.Matrix4().copy(parent.matrixWorld).invert();
    const localMatrix = parentWorldMatrixInverse.multiply(bodyWorldMatrix);
    localMatrix.decompose(group.position, group.quaternion, group.scale);

    parent.add(group);
    parent.updateMatrixWorld(true);

    if (body.geoms) {
      body.geoms.forEach(geom => {
        if(geom.group !== 3) {
          const mesh = createGeomRelative(geom);
          group.add(mesh);
        }
      });
    }

    if (body.bodies) {
      body.bodies.forEach(b => renderBodyAbsolute(b, group));
    }
  };

  const createGeomRelative = (geom: Geom): three.Mesh | three.Group => {
    let geometry: three.BufferGeometry;
    let object: three.Mesh | three.Group;
    
    if (geom.geom_t === 'Mesh' && geom.mesh_obj) {
      const loader = new OBJLoader();
      object = loader.parse(geom.mesh_obj);
      if (geom.pos) {
        object.position.set(geom.pos[0], geom.pos[1], geom.pos[2]);
      }
      if (geom.quat) {
        object.quaternion.set(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]);
      } else if (geom.euler) {
        const euler = new three.Euler(
          three.MathUtils.degToRad(geom.euler[0]),
          three.MathUtils.degToRad(geom.euler[1]),
          three.MathUtils.degToRad(geom.euler[2])
        );
        object.setRotationFromEuler(euler);
      }
      const color = new three.Color(0x198470);
      object.traverse((child) => {
        if (child instanceof three.Mesh) {
          child.material = new three.MeshStandardMaterial({ color: color });
        }
      });
      return object;
    }

    switch (geom.geom_t) {
      case "Box":
        geometry = new three.BoxGeometry(geom.size[0] * 2, geom.size[1] * 2, geom.size[2] * 2);
        break;
      case 'Sphere':
        geometry = new three.SphereGeometry(geom.size[0]);
        break;
      case 'Cylinder':
      case 'Capsule': 
        geometry = new three.CylinderGeometry(geom.size[0], geom.size[0], geom.size[1] * 2);
        break;
      case 'Plane':
        geometry = new three.BoxGeometry(geom.size[0] || 100, geom.size[1] || 100, 0.01);
        break;
      default:
        geometry = new three.BoxGeometry(1, 1, 1);
        break;
    }

    const color = new three.Color(0x198470);
    const material = new three.MeshStandardMaterial({ color: color });
    const mesh = new three.Mesh(geometry, material);
    
    if (geom.pos) {
      mesh.position.set(geom.pos[0], geom.pos[1], geom.pos[2]);
    }
    if (geom.quat) {
      mesh.quaternion.set(geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]);
    } else if (geom.euler) {
      const euler = new three.Euler(
        three.MathUtils.degToRad(geom.euler[0]),
        three.MathUtils.degToRad(geom.euler[1]),
        three.MathUtils.degToRad(geom.euler[2])
      );
      mesh.setRotationFromEuler(euler);
    }
    
    return mesh;
  };
  
  const updateBodyAndChildren = (bodies: Body[], update: ModelForward) => {
    for (const body of bodies) {
        if (body.body_id) {
            const bodyObject = bodyObjectMap.get(body.body_id);
            if (bodyObject && bodyObject.parent) {
                const posIndex = body.body_id * 3;
                const quatIndex = body.body_id * 4;

                if (posIndex + 2 < update.xpos.length && quatIndex + 3 < update.xquat.length) {
                    const newPos = new three.Vector3().fromArray(update.xpos, posIndex);
                    const newQuat = new three.Quaternion(
                        update.xquat[quatIndex + 1],
                        update.xquat[quatIndex + 2],
                        update.xquat[quatIndex + 3],
                        update.xquat[quatIndex]
                    );

                    const bodyWorldMatrix = new three.Matrix4().compose(newPos, newQuat, new three.Vector3(1, 1, 1));
                    const parentWorldMatrixInverse = new three.Matrix4().copy(bodyObject.parent.matrixWorld).invert();
                    const localMatrix = new three.Matrix4().multiplyMatrices(parentWorldMatrixInverse, bodyWorldMatrix);
                    
                    localMatrix.decompose(bodyObject.position, bodyObject.quaternion, bodyObject.scale);
                    
                    // Crucially, update this body's world matrix so children can use the correct value
                    bodyObject.updateMatrixWorld(true);
                }
            }
        }
        if (body.bodies) {
            updateBodyAndChildren(body.bodies, update);
        }
    }
  }

  const updateSimul = (update: ModelForward) => {
    if (!update || !worldBody) return;
    updateBodyAndChildren(worldBody.bodies, update);
  };

  const renderWorld = (worldBody: WorldBody) => {
    if(!scene) return;

    while(scene.children.length > 0) {
      scene.remove(scene.children[0]);
    }
    bodyObjectMap.clear();
    
    const ambientLight = new three.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new three.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    if(worldBody && worldBody.bodies) {
      worldBody.bodies.forEach((b: Body) => renderBodyAbsolute(b, scene));
    }
  }

  onMount(() => {
    three.Object3D.DEFAULT_UP.set(0, 0, 1);
    scene = new three.Scene();
    scene.background = new three.Color(0xFCB603);
    camera = new three.PerspectiveCamera(20, width / height, 0.5, 1000);
    camera.up.set(0, 0, 1);
    renderer = new three.WebGLRenderer({ canvas: canvas, antialias: true });
    cameraController = new OrbitControls(camera,canvas);
    resizeRenderer();
    camera.position.z = 5;
    cameraController.update();
    const resizeObserver = new ResizeObserver(() => {
      resizeRenderer();
    });
    resizeObserver.observe(viewportEl);

    function animate() {
      requestAnimationFrame(animate);
      cameraController.update();
      renderer.render(scene, camera);
    }

    animate();

    return () => {
      resizeObserver.disconnect();
      cameraController.dispose();
      renderer.dispose();
    };
  });

  $effect(() => {
    if (worldBody) {
      renderWorld(worldBody);
    }
  });

  $effect(() => {
    if (simulUpdateData) {
      updateSimul(simulUpdateData);
    }
  });
  
</script>

<div
  class="viewport"
  bind:this={viewportEl}
  style="width: 100%; height: {height}px;"
>
    <canvas 
        bind:this={canvas} 
        style="width: 100%; height: 100%; touch-action: none;"
    ></canvas>
</div>
