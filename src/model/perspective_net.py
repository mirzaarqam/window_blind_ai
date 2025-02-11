from controlnet_aux import HEDdetector
from diffusers import ControlNetModel


class PerspectiveNetwork:
    def __init__(self):
        self.hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-hed"
        )

    def detect_edges(self, image):
        return self.hed(image)