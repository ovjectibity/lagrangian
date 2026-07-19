import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import os
from PIL import Image
import numpy as np

MJCF_PATH = os.environ.get("MJCF_PATH", "/opt/lagrangian/trossen_vx300s/vx300s.xml")

def run_render():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mjx_model = mjx.put_model(model,impl="warp")
    mjx_data = mjx.make_data(model,impl="warp")
    ctx = mjx.create_render_context(model,1)
    # print(ctx.keys())
    rgb, _ = mjx.render(mjx_model,mjx_data,ctx.pytree())
    for i in range(mjx_model.ncam): 
        rgb_data = mjx.get_rgb(ctx.pytree(),i,rgb)
        rgb_data = rgb_data * 255
        image = Image.fromarray(np.asarray(rgb_data[0]).astype(np.uint8), mode="RGB")
        image.save(f"/opt/lagrangian/trossen_vx300s/output_image_{i}.png")
    return

def main(): 
    run_render()
    return

if __name__ == "__main__":
    main()