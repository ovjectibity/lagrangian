<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Viewport from './lib/viewport.svelte';
  import MenuBar from './lib/menubar.svelte';
  import type { WorldBody } from './common/types';

  let websocket: WebSocket | null = null;
  let worldBody: WorldBody | undefined = $state(undefined);

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
        const data = JSON.parse(event.data);
        if (data && data.set_model && data.set_model.bodies) { // Check for set_model field and its bodies
          console.log("Got this message type: ",data.msg_type);
          worldBody = data.set_model as WorldBody;
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

    // // Initial request for model if websocket is ready
    // if (websocket.readyState === WebSocket.OPEN) {
    //     websocket.send("get_test_model");
    // }

    return () => {
      if (websocket) {
        websocket.close();
      }
      window.removeEventListener('keydown', handleKeyDown);
    };
  });

  onDestroy(() => {
    // onMount cleanup function is sufficient for onDestroy tasks in Svelte 5 runes.
    // However, explicitly closing for clarity if somehow not handled by onMount's return.
    if (websocket) {
        websocket.close();
    }
    window.removeEventListener('keydown', handleKeyDown);
  });
</script>

<main>
  <MenuBar />
  <div>
    <Viewport width={800} height={500} worldBody={worldBody}/>
  </div>
</main>