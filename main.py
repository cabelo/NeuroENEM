from gradio_helper import make_demo
from transformers import AutoProcessor, TextStreamer
from PIL import Image
from io import BytesIO

pathM ="Gemma-3-Gaia-PT-BR-4b-it-int8-ov"
from optimum.intel.openvino import OVModelForVisualCausalLM
model = OVModelForVisualCausalLM.from_pretrained(pathM, device="CPU")
processor = AutoProcessor.from_pretrained(pathM)
neuroEnem = make_demo(model, processor)

neuroEnem.launch(share=True,debug=True)
