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

    websocket.onopen = () => {
      console.log("WebSocket connection established.");
    };

    websocket.onmessage = (event) => {
      console.log("Received message:", event.data);
      try {
        const clientMessage: ClientMessage = JSON.parse(event.data);
        if ("SetModelMessage" in clientMessage) {
          worldBody = clientMessage.SetModelMessage.set_model;
          simulUpdateData = undefined; // Clear simulation data on new model load
        } else if ("LoadModelMessage" in clientMessage) {
          worldBody = clientMessage.LoadModelMessage.model;
          simulUpdateData = undefined; // Clear simulation data on new model load
        } else if ("SimulUpdateMessage" in clientMessage) {
          // Assuming only one update per message for now, can be extended if needed
          simulUpdateData = clientMessage.SimulUpdateMessage.updates[0];
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message as JSON:", e);
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