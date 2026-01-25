<script lang="ts">
  import * as three from 'three';
  import { onMount } from 'svelte';
  import type { WorldBody, Body, Geom } from '../common/types';
  import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

  const {
    width = 500, 
    height = 300,
    worldBody = undefined
  } = $props<{ width: number; height: number, worldBody?: WorldBody}>();

  let canvas: HTMLCanvasElement;
  let scene: three.Scene;
  let camera: three.PerspectiveCamera;
  let renderer: three.WebGLRenderer;

  let isDragging = false;
  let previousMousePosition = {
    x: 0,
    y: 0
  };
  const rotationSpeed = 0.005;

  const onMouseDown = (event: MouseEvent) => {
    console.log("Mouse down on viewport");
    isDragging = true;
    previousMousePosition = {
      x: event.clientX,
      y: event.clientY
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  const onMouseMove = (event: MouseEvent) => {
    if (!isDragging || !camera) return;

    console.log("Mouse move on viewport");
    const deltaX = event.clientX - previousMousePosition.x;
    const deltaY = event.clientY - previousMousePosition.y;

    // Get the vector perpendicular to the camera's direction and up vectors (the "right" vector)
    const right = new three.Vector3().
      crossVectors(camera.up, new three.Vector3().
      subVectors(new three.Vector3(0,0,0), camera.position).
      normalize());

    // Rotate the camera's position around the world's Y axis
    camera.position.applyAxisAngle(
      new three.Vector3(0, 1, 0), -deltaX * rotationSpeed);
    // Rotate the camera's position around its local "right" axis
    camera.position.applyAxisAngle(right, -deltaY * rotationSpeed);
    
    // Point the camera back at the origin
    camera.lookAt(0, 0, 0);

    previousMousePosition = {
      x: event.clientX,
      y: event.clientY
    };
  };

  const onMouseUp = () => {
    console.log("Mouse up on viewport");
    isDragging = false;
    window.removeEventListener('mousemove', onMouseMove);
    window.removeEventListener('mouseup', onMouseUp);
  };

  const renderBody = (body: Body, parent: three.Object3D) => {
    console.log("Rendering body:", body.name, body.pos);
    const group = new three.Group();
    if (body.name) {
      group.name = body.name;
    }
    if (body.pos) {
      group.position.set(body.pos[0], body.pos[1], body.pos[2]);
    }
    if (body.quat) {
      // MuJoCo is (w, x, y, z), Three is (x, y, z, w)
      group.quaternion.set(body.quat[1], body.quat[2], body.quat[3], body.quat[0]);
    } else if (body.euler) {
      const euler = new three.Euler(
        three.MathUtils.degToRad(body.euler[0]),
        three.MathUtils.degToRad(body.euler[1]),
        three.MathUtils.degToRad(body.euler[2])
      );
      group.setRotationFromEuler(euler);
    }
    parent.add(group);

    if (body.geoms) {
      body.geoms.forEach(geom => {
        console.log("  Rendering geom:", geom.name, geom.geomT, geom.pos);
        const mesh = createGeom(geom);
        group.add(mesh);
      });
    }

    if (body.bodies) {
      body.bodies.forEach(b => renderBody(b, group));
    }
  };

  const createGeom = (geom: Geom): three.Mesh | three.Group => {
    let geometry: three.BufferGeometry;
    
    if (geom.geomT === 'mesh' && geom.mesh_obj) {
      const loader = new OBJLoader();
      const object = loader.parse(geom.mesh_obj);
      // The loader returns a Group. We can apply transformations to the group.
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
      // Apply color to all meshes in the group
      // const color = geom.rgba ? new three.Color(geom.rgba[0], geom.rgba[1], geom.rgba[2]) : new three.Color(0xcccccc);
      const color = new three.Color(0x198470);
      object.traverse((child) => {
        if (child instanceof three.Mesh) {
          child.material = new three.MeshStandardMaterial({ color: color });
        }
      });
      return object;
    }

    switch (geom.geomT) {
      case '_box':
        geometry = new three.BoxGeometry(geom.size[0] * 2, geom.size[1] * 2, geom.size[2] * 2);
        break;
      case 'sphere':
        geometry = new three.SphereGeometry(geom.size[0]);
        break;
      case 'cylinder':
      case 'capsule': // Approximating capsule as a cylinder
        geometry = new three.CylinderGeometry(geom.size[0], geom.size[0], geom.size[1] * 2);
        break;
      case 'plane':
        // Mujoco plane is infinite, three.js plane is not. Representing as a large thin box.
        geometry = new three.BoxGeometry(geom.size[0] || 100, geom.size[1] || 100, 0.01);
        break;
      case 'mesh':
        // Fallback for mesh if mesh_obj is not provided
        geometry = new three.BoxGeometry(1, 1, 1);
        break;
      default:
        geometry = new three.BoxGeometry(1, 1, 1);
        break;
    }

    // const color = geom.rgba ? new three.Color(geom.rgba[0], geom.rgba[1], geom.rgba[2]) : new three.Color(0xcccccc);
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

  const renderWorld = () => {
    if (!scene) return;
    // Clear previous objects
    while(scene.children.length > 0){
      scene.remove(scene.children[0]);
    }
    
    // Add lights
    const ambientLight = new three.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new three.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    if (worldBody && worldBody.bodies) {
      worldBody.bodies.forEach((b: Body) => renderBody(b, scene));
    }
  }

  onMount(() => {
    scene = new three.Scene();
    scene.background = new three.Color(0xFCB603);
    camera = new three.PerspectiveCamera(75, width / height, 0.1, 1000);
    renderer = new three.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(width, height);

    camera.position.z = 5;

    // Add a test sphere directly
    const geometry = new three.SphereGeometry(0.5);
    const material = new three.MeshStandardMaterial({ color: 0x00ff00 }); // Green color
    const sphere = new three.Mesh(geometry, material);
    sphere.position.set(0, 0, 0);
    scene.add(sphere);

    function animate() {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }

    animate();
    // renderWorld(); // We don't call this directly anymore, it's handled by $effect
  });
  
  $effect(() => {
    if (worldBody) {
      renderWorld();
    }
  });
</script>

<div class="viewport" style="width: {width}px; height: {height}px;">
    <canvas bind:this={canvas} width={width} height={height} onmousedown={onMouseDown}></canvas>
</div>
