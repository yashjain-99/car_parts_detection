try:
    from models.common import DetectMultiBackend
except Exception as e:
    import zipfile
    with zipfile.ZipFile("models_zip.zip","r") as zip_ref:
        zip_ref.extractall("models")
    with zipfile.ZipFile("utils_zip.zip","r") as zip_ref:
        zip_ref.extractall("utils")
from models.common import DetectMultiBackend

class GetModels:
    def __init__(self, model_onxx_path, device):
        self.carparts = DetectMultiBackend(
            model_onxx_path, 
            device=device, 
            dnn=False, 
            data=None, 
            fp16=False)

    def get_models(self):
        return self.carparts