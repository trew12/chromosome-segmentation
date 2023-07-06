from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from hydra import compose, initialize
import os
from starlette.responses import FileResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import shutil
import uvicorn


app = FastAPI()
app.mount("/app", StaticFiles(directory="app"), name="app")
with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="segment_config")


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    return RedirectResponse("/")


@app.get("/")
async def main():
    with open("app/main.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/predict")
async def predict(threshold: float = Form(...), confidence: float = Form(...), image: UploadFile = File(...)):
    save_folder = 'user_input'
    file_path = os.path.join(save_folder, 'input_image.png')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    os.system(f'python segment.py -cn segment_config img_path={file_path} mask_threshold={threshold} confidence={confidence}')
    if os.path.isfile(cfg.save_path):
        return FileResponse(cfg.save_path)
    else:
        return None


if __name__ == "__main__":
    uvicorn.run('app:app', host="0.0.0.0", port=8000)
