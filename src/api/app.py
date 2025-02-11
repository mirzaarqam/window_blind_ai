from fastapi import FastAPI, UploadFile, File
from inference import process_images

app = FastAPI()

@app.post("/integrate-blinds")
async def integrate_blinds(
    window: UploadFile = File(...),
    blind: UploadFile = File(...)
):
    result = process_images(
        await window.read(),
        await blind.read()
    )
    return {"result": result}