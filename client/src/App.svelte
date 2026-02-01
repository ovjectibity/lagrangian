<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Viewport from './lib/viewport.svelte';
  import MenuBar from './lib/menubar.svelte';
  import type { WorldBody, ClientMessage, ModelForward } from './common/types';

  let websocket: WebSocket | null = null;
  let worldBody: WorldBody | undefined = $state(undefined);
  let simulUpdateData: ModelForward | undefined = $state(undefined);

  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'l' && websocket && websocket.readyState === WebSocket.OPEN) {
      console.log("Sending 'get_test_model' message...");
      websocket.send("get_test_model");
    } else if (event.key === 's' && websocket && websocket.readyState === WebSocket.OPEN) {
      console.log("Initiating simualtion loop...");
      websocket.send("run_simul");
    } else if (event.key === 'g' && websocket && websocket.readyState === WebSocket.OPEN) {
      console.log("Getting test model 2...");
      websocket.send("get_test_model_2");
    } else if (event.key === 'r' && websocket && websocket.readyState === WebSocket.OPEN) {
      console.log("Getting test model 3...");
      websocket.send("get_test_model_3");
    }
  };

  onMount(() => {
    websocket = new WebSocket("ws://localhost:8000/api/viewport");
    websocket.binaryType = 'arraybuffer'; // Important for receiving binary data

    websocket.onopen = () => {
      console.log("WebSocket connection established.");
    };

    websocket.onmessage = (event) => {
      if (typeof event.data === 'string') {
        // Handle JSON messages
        try {
          const clientMessage: ClientMessage = JSON.parse(event.data);
          if ("SetModelMessage" in clientMessage) {
            worldBody = clientMessage.SetModelMessage.set_model;
            simulUpdateData = undefined; // Clear simulation data on new model load
          } else if ("LoadModelMessage" in clientMessage) {
            worldBody = clientMessage.LoadModelMessage.model;
            simulUpdateData = undefined; // Clear simulation data on new model load
          }
        } catch (e) {
          console.error("Failed to parse WebSocket JSON message:", e);
        }
      } else if (event.data instanceof ArrayBuffer) {
        // Handle binary messages
        const buffer = new DataView(event.data);
        let offset = 0;

        const messageType = buffer.getUint8(offset);
        offset += 1;

        if (messageType === 7) { // Hardcoded type for SimulUpdateMessage
          const idLen = buffer.getUint16(offset, true); // true for little-endian
          offset += 2;
          const idBytes = new Uint8Array(event.data, offset, idLen);
          const id = new TextDecoder().decode(idBytes);
          offset += idLen;

          const nbody = buffer.getUint16(offset, true);
          offset += 2;

          const xpos: number[] = [];
          for (let i = 0; i < nbody * 3; i++) {
            xpos.push(buffer.getFloat64(offset, true));
            offset += 8;
          }

          const xquat: number[] = [];
          for (let i = 0; i < nbody * 4; i++) {
            xquat.push(buffer.getFloat64(offset, true));
            offset += 8;
          }

          simulUpdateData = { xpos, xquat };
        } else {
          console.warn("Received unknown binary message type:", messageType);
        }
      }
    };

    websocket.onclose = (event) => {
      console.log("WebSocket connection closed:", event);
      websocket = null;
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      if (websocket) {
        websocket.close();
      }
      window.removeEventListener('keydown', handleKeyDown);
    };
  });

  onDestroy(() => {
    if (websocket) {
        websocket.close();
    }
    window.removeEventListener('keydown', handleKeyDown);
  });
</script>

<main>
  <MenuBar />
  <div>
    <Viewport width={800} height={500} worldBody={worldBody} simulUpdateData={simulUpdateData}/>
  </div>
</main>